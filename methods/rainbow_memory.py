import logging
import copy

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.datasets import ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from utils.memory import MemoryBatchSampler, MemoryOrderedSampler
logger = logging.getLogger()

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class RM(ER):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sched_name = "const"
        self.batch_size = kwargs["batchsize"]
        self.memory_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.data_cnt = 0

    def setup_distributed_dataset(self):
        super(RM, self).setup_distributed_dataset()
        self.loss_update_dataset = self.datasets[self.dataset](root=self.data_dir, train=True, download=True,
                                     transform=transforms.ToTensor())

    def online_step(self, images, labels, idx):
        # image, label = sample
        self.add_new_class(labels)
        self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=0, pin_memory=True)
        self.memory_provider     = iter(self.memory_dataloader)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter) * self.temp_batchsize * self.world_size):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels)
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in range(len(memory_labels)):
                memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        x = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def add_new_class(self, class_name):
        super(RM,self).add_new_class(class_name)
        # self.reset_opt()

    def update_memory(self, index, label):
        # Update memory
        if self.distributed:
            index = torch.cat(self.all_gather(index.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            index = index.cpu()
            label = label.cpu()
        
        for x, y in zip(index, label):
            if len(self.memory) >= self.memory_size:
                label_frequency = copy.deepcopy(self.memory.cls_count)
                label_frequency[self.exposed_classes.index(y.item())] += 1
                cls_to_replace = torch.argmax(label_frequency)
                cand_idx = (self.memory.labels == self.memory.cls_list[cls_to_replace]).nonzero().squeeze()
                idx_to_replace = cand_idx[torch.randint(0, len(cand_idx), (1,))]
                self.memory.replace_data([x,y], int(idx_to_replace))
            else:
                self.memory.replace_data([x,y])

    def online_before_task(self, cur_iter):
        # self.reset_opt()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

    def online_after_task(self, cur_iter):
        self.model.train()
        # self.optimizer = select_optimizer(self.opt_name,self.lr,self.model)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
        #     )
        self.online_memory_train(
            cur_iter=cur_iter,
            n_epoch=self.memory_epoch,
            batch_size=self.batch_size,
        )

    def online_memory_train(self, cur_iter, n_epoch, batch_size):
        if self.dataset == 'imagenet':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30, 60, 80, 90], gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )
        self.memory_sampler = MemoryOrderedSampler(self.memory, self.batchsize, 1)
        self.memory_dataloader = DataLoader(self.train_dataset, batch_size=self.batchsize, sampler=self.memory_sampler, num_workers=4, pin_memory=True)
        for epoch in range(n_epoch):
            self.model.train()
            self.memory_sampler = MemoryOrderedSampler(self.memory, self.batchsize, len(self.memory) // self.batchsize)
            self.memory_dataloader = DataLoader(self.loss_update_dataset, batch_size=self.batchsize, sampler=self.memory_sampler, num_workers=4, pin_memory=True)
            
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            total_loss, correct, num_data = 0.0, 0.0, 0.0

            for n_batches, (image, label) in enumerate(self.memory_dataloader):

                x = image.to(self.device)
                y = label.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                x = self.train_transform(x)

                self.optimizer.zero_grad()

                logit, loss = self.model_forward(x, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)
                
            n_batches = len(self.memory_dataloader)
            train_loss, train_acc = total_loss / n_batches, correct / num_data
            print(f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | lr {self.optimizer.param_groups[0]['lr']:.4f}")

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling
        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric="vr_randaug")

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class

        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            try:
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )
            except:
                ret += (
                    sample_df[~sample_df.filepath.isin(pd.DataFrame(ret).filepath)]
                        .sample(n=num_rest_slots)
                        .to_dict(orient="records")
                )

        try:
            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        except:
            num_dups = pd.DataFrame(ret).filepath.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset=self.dataset, transform=infer_transform, data_dir=self.data_dir
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12
        elif uncert_metric == "vr_autoaug":
            transform_cands = [select_autoaugment(self.dataset)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learned_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            try:
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )
            except:
                ret += (
                    sample_df[~sample_df.filepath.isin(pd.DataFrame(ret).filepath)]
                        .sample(n=num_rest_slots)
                        .to_dict(orient="records")
                )

        try:
            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        except:
            num_dups = pd.DataFrame(ret).filepath.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret