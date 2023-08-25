import logging
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.memory import MemoryBatchSampler, MemoryOrderedSampler,BatchSampler

from methods._trainer import _Trainer
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class RM(ER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sched_name = "const"
        self.batch_size = kwargs["batchsize"]
        self.memory_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.data_cnt = 0
        
        self.cur_iter = None

    def setup_distributed_dataset(self):
        super(RM, self).setup_distributed_dataset()
        self.loss_update_dataset = self.datasets[self.dataset](root=self.data_dir, train=True, download=True,
                                     transform=transforms.Compose([transforms.Resize((self.inp_size,self.inp_size)),transforms.ToTensor()]))

    def online_step(self, images, labels, idx):
        #* images -> image list
        #* labels -> label list
        # image, label = sample
        self.add_new_class(labels[0])
        # self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        # self.memory_dataloader = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=0, pin_memory=True)
        # self.memory_provider = iter(self.memory_dataloader)
        # train with augmented batches
        # print("[Online-Step]")
        # print(images)
        # print(len(images))
        # print(images.shape)
        # print(labels)
        # print(labels[0])
        _loss, _acc, _iter = 0.0, 0.0, 0
        for image, label in zip(images, labels):
            loss, acc = self.online_train([image.clone(), label.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels[0])
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        # if len(self.memory) > 0 and self.memory_batchsize > 0:
            # memory_batchsize = min(self.memory_batchsize, len(self.memory))
            # memory_images, memory_labels = self.memory.get_batch(memory_batchsize)
            # memory_images, memory_labels = next(self.memory_provider)
            # x = torch.cat([x, memory_images], dim=0)
            # y = torch.cat([y, memory_labels], dim=0)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        # x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])
        # x = torch.cat([self.train_transform(_x).unsqueeze(0) for _x in x])

        x = x.to(self.device)
        y = y.to(self.device)

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
        self.reset_opt()

    def update_memory(self, sample, label):
        #! normal balanced memory sampling
        #! =====================================================================================
        # Update memory
        # if self.distributed:
        #     sample = torch.cat(self.all_gather(sample.to(self.device)))
        #     label  = torch.cat(self.all_gather(label.to(self.device)))
        #     sample = sample.cpu()
        #     label  = label.cpu()
        
        # for x, y in zip(sample, label):
        #     if len(self.memory) >= self.memory_size:
        #         label_frequency = copy.deepcopy(self.memory.cls_count)
        #         label_frequency[self.exposed_classes.index(y.item())] += 1
        #         cls_to_replace = torch.argmax(label_frequency)
        #         idx_to_replace = cls_to_replace[torch.randint(0, len(cls_to_replace), (1,))]
        #         self.memory.replace_data([x,y], idx_to_replace)
        #     else:
        #         self.memory.replace_data([x,y])
        #! =====================================================================================
        
        #? uncertainty_based sampling
        #? -------------------------------------------------------------
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label  = torch.cat(self.all_gather(label.to(self.device)))
            # self.exposed_classes  = torch.cat(self.all_gather(torch.tensor(self.exposed_classes).to(self.device)))
            # self.exposed_classes = self.exposed_classes.tolist()
            sample = sample.cpu()
            label  = label.cpu()
        
        # for x,y in zip(sample,label):
        stored_x =[]
        stored_y =[]
        # print("[Update Memory] Memory sample num:",len(self.memory))
        if self.cur_iter == 0:
            for x, y in zip(sample, label):
                
                if len(self.memory) >= self.memory_size:
                    # print("memory buffer Full, need to exchange")
                    label_frequency = copy.deepcopy(self.memory.cls_count)
                    label_frequency[self.exposed_classes.index(y.item())] += 1
                    cls_to_replace = torch.argmax(label_frequency)
                    cand_idx = (self.memory.labels == self.memory.cls_list[cls_to_replace]).nonzero().squeeze()
                    idx_to_replace = cand_idx[torch.randint(0, len(cand_idx), (1,))]
                    self.memory.replace_data([x.float(),y.float()], idx_to_replace)
                else:
                    self.memory.replace_data([x,y])
        else:
            # print("save data into memory")
            #todo --------------------------------------
            if len(self.memory) >= self.memory_size:
                
                for x, y in zip(sample, label):
                    # print(x.shape)
                    stored_x.append(x)
                    stored_y.append(y)
                
                #todo --------------------------------------
                #todo uncertainty_sampling
                # self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
                # self.memory_dataloader = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=0, pin_memory=True)
                if len(self.memory) > 0:
                    for mem_x, mem_y in zip(self.memory.memory,self.memory.labels):
                        stored_x.append(mem_x)
                        stored_y.append(mem_y)
                # print(len(stored_x))
                # print(stored_x)
                stored_idx= torch.randperm(len(stored_x))
                stored_x = torch.tensor(stored_x)[stored_idx]
                stored_y = torch.tensor(stored_y)[stored_idx]
                print(f"stored_x:{stored_x.shape}")
                print(f"stored_y:{stored_y.shape}")
                print(stored_y)
                # self.uncert_table=torch.zeros((stored_x.shape[0],))
                self.uncert_all_info = {}
                # print("start uncertainty_sampling process")
                self.memory_list = self.uncertainty_sampling(
                                    [stored_x,stored_y],
                                    num_class=len(self.exposed_classes))
                # print("Done uncertainty_sampling process")
                # assert len(self.memory_list) <= self.memory_size
                
                # print("uncertainty Sampling")
                # print(self.memory_list)
                sample_idx,label = self.memory_list
                print()
                print("sample_idx")
                print(sample_idx)
                print(len(sample_idx))
                print("label")
                print(label)
                print(len(label))
                print()
                # sample_idx = sample_idx[:self.memory_size]
                # label = label[:self.memory_size]
                print()
                print("exposed CLass")
                print(self.exposed_classes)
                print()
                for idx, (x,y) in enumerate(zip(sample_idx,label)):
                    x = torch.tensor([int(x)],dtype=torch.float)
                    # y = torch.tensor([int(y)],dtype=torch.float)
                    # print('x',x)
                    # print(type(x))
                    # print('y_before',y)
                    # y = self.exposed_classes[y]
                    # print('y_after',y)
                    # print(type(y))
                    
                    # print(self.memory.cls_count[(self.memory.cls_list == label).nonzero().squeeze()])
                    # print()
                    # print(self.memory.cls_count[(self.memory.cls_list == label)])
                    self.memory.replace_data([x,y],idx)
            else:
                for x, y in zip(sample, label):
                    self.memory.replace_data([x,y])
        #? -------------------------------------------------------------
        
        

    def online_before_task(self, cur_iter):
        self.cur_iter = cur_iter
        self.reset_opt()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

    def online_after_task(self, cur_iter):
        # print("start online_after_train!")
        self.model.train()
        # self.reset_opt()
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
        #!-------------------------------------------------------------------------------
        # print("Memory Sample Num:",len(self.memory))
        # self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        # self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.memory_sampler, num_workers=0, pin_memory=True)
        # self.memory_provider     = iter(self.memory_dataloader)
        num_iter = math.ceil(len(self.memory)/batch_size)
        #!-------------------------------------------------------------------------------
        memory_sampler = MemoryOrderedSampler(self.memory, self.batchsize, 1)
        memory_dataloader   = DataLoader(self.train_dataset, batch_size=batch_size, sampler=memory_sampler, num_workers=4, pin_memory=True)
        for epoch in range(n_epoch):
            self.model.train()
            
            # self.memory_provider     = iter(self.memory_dataloader)
            
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            total_loss, correct, num_data = 0.0, 0.0, 0.0
            mem_config = torch.zeros((len(self.exposed_classes),)).to(self.device)
            for i, (memory_images, memory_labels) in enumerate(memory_dataloader):
            #* for n_iter in range(num_iter):
            #*     memory_images, memory_labels = next(self.memory_provider)
                # try:
                #    memory_images, memory_labels = next(self.memory_provider)
                # except StopIteration:
                #     batch_iterator = iter(self.memory_dataloader)
                #     memory_images, memory_labels = next(batch_iterator)
                # idx = torch.randperm(len(memory_labels))
                # x = memory_images[idx].to(self.device)
                # y = memory_labels[idx].to(self.device)
                x = memory_images.to(self.device)
                y = memory_labels.to(self.device)
                print("[Memory] Label")
                print(memory_labels)
                print("y")
                print(y)
                
                # print("[Memory Train] before y:",y)
                l_idx=[]
                for j in range(len(y)):
                    if y[j] not in self.exposed_classes:
                        print('y[j]:',y[j])
                        print("j:",j)
                        print(self.exposed_classes)
                        # raise ValueError("What the Fuck!")
                    y[j] = self.exposed_classes.index(y[j].item())
                    
                        
                # if i%10 ==0:
                    # print(f'iter:{i} shape:{x.shape} labels:{y}')
                mem_config[y] +=1


                self.optimizer.zero_grad()

                logit, loss = self.model_forward(x, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                # if self.use_amp:
                self.scaler.scale(loss).backward()
                # self.scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # else:
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                #     self.optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)
                # print(f"Task {cur_iter} | iter {i + 1}/{num_iter} | train_loss {total_loss/(i+1):.4f} | train_acc {correct/num_data:.4f} | ")
            # n_batches = len(DataLoader)
            train_loss, train_acc = total_loss / num_iter, correct / num_data
            if epoch%10 ==0:
                print(f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | lr {self.optimizer.param_groups[0]['lr']:.4f}")
                # print(f"lr {self.optimizer.param_groups[0]['lr']:.4f}")
            # print(f"memory config:{mem_config} sum:{mem_config.sum()}")

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [images , labels ]
        """
        # print("start montecarlo process")
        self.montecarlo(samples, uncert_metric="vr_randaug")
        # print("Done montecarlo process")
        # sample_df = pd.DataFrame(samples)
        # sample_df = pd.DataFrame(self.uncert_all_info)
        mem_per_cls = self.memory_size // num_class
        print("mem_per_cls:",mem_per_cls)
        # print()
        # print("Data Frame!!")
        # print(sample_df.loc[0])
        

        ret_idx = []
        ret_label=[]
        for i in (self.exposed_classes):
            print("Sampling Class:",i)
            # cls_df = sample_df[sample_df["label"] == i]
            cls_list=[]
            for key in self.uncert_all_info.keys():
                if self.uncert_all_info[key]['label'] == i:
                    cls_list.append(key)
            print("cls_list:",cls_list)
                    
            if len(cls_list) <= mem_per_cls:
                # ret += cls_df.to_dict(orient="records")
                ret_idx += cls_list
                # for _ in range(len(cls_list))
                    # ret_label.append(i)
            else:
                jump_idx = len(cls_list) // mem_per_cls
                uncert_list=[]
                for key in cls_list:
                    uncert_list.append(self.uncert_all_info[key]['uncertainty'])
                uncert_list = np.array(uncert_list)
                print('uncert_list',uncert_list)
                uncertain_samples = uncert_list.argsort()[::jump_idx]
                
                
                # uncertain_samples = cls_list.sort_values(by="uncertainty")[::jump_idx]
                
                ret_idx += torch.tensor(uncertain_samples)[:mem_per_cls]
            # print("len(ret_idx):",len(ret_idx))
            for _ in range(len(cls_list)):
                ret_label.append(i)
                # for _ in range(len())

        # num_rest_slots = self.memory_size - len(ret)
        # if num_rest_slots > 0:
        #     # logger.warning("Fill the unused slots by breaking the equilibrium.")
        #     try:
        #         ret += (
        #             sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
        #             .sample(n=num_rest_slots)
        #             .to_dict(orient="records")
        #         )
        #     except:
        #         ret += (
        #             sample_df[~sample_df.filepath.isin(pd.DataFrame(ret).filepath)]
        #                 .sample(n=num_rest_slots)
        #                 .to_dict(orient="records")
        #         )

        # try:
        #     num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        # except:
        #     num_dups = pd.DataFrame(ret).filepath.duplicated().sum()
        # if num_dups > 0:
        #     logger.warning(f"Duplicated samples in memory: {num_dups}")
        print("ret_idx:",ret_idx);print()
        print("ret_label:",ret_label);print(len(ret_label))
        print()
        print("Exposed Class:",self.exposed_classes)
        
        # for j in range(len(ret_label)):
        #     class_idx = ret_label[j]
        #     ret_label[j] = self.exposed_classes[class_idx]
            
        # print("\nret_label--After:",ret_label);print()
        
        return [ret_idx, ret_label]

    def _compute_uncert(self, infer_samples, infer_transform, uncert_name):
        #* sample_info -> dict type for a sample 
        batch_size = self.batch_size
        # infer_df = pd.DataFrame(infer_list)
        # infer_dataset = ImageDataset(
        #     infer_df, dataset=self.dataset, transform=infer_transform, data_dir=self.data_dir
        # )
        # infer_loader = DataLoader(
        #     infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        # )
        # inputs,labels = infer_samples
        # infer_dataset = ImageDataset(
        #     infer_samples, transform=infer_transform
        # )
        infer_x,infer_y = infer_samples
        mem_cand  = BatchSampler(infer_x, self.memory_batchsize, 1)
        infer_loader = DataLoader(
            self.train_dataset, shuffle=False, batch_size=batch_size,sampler=mem_cand, num_workers=0
        )
        # self.uncert_info = {}
        self.model.eval()
        with torch.no_grad():
            for n_batch, (x,y) in enumerate(infer_loader):
                # x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()[:,:len(self.exposed_classes)]
                # if n_batch ==0:
                #     print('logit',logit.shape)
                # break

                for i, cert_value in enumerate(logit):
                    single_idx = str(batch_size * n_batch + i)
                    # sample_info[str(batch_size * n_batch + i)] = infer_dataset.images[batch_size * n_batch + i]
                    if single_idx not in self.uncert_all_info.keys():
                        self.uncert_all_info[single_idx]={}
                    else:
                        pass
                    # self.uncert_all_info[single_idx]['sample'] = infer_dataset.images[single_idx]
                    # self.uncert_all_info[single_idx]['label'] = infer_dataset.labels[single_idx]
                    # print("uncert_all_info")
                    # print(self.uncert_all_info)
                    self.uncert_all_info[single_idx]['sample'] = infer_x[batch_size * n_batch + i]
                    self.uncert_all_info[single_idx]['label'] = infer_y[batch_size * n_batch + i]
                    self.uncert_all_info[single_idx][uncert_name] = 1 - cert_value
        # return sample_info

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        # print(f"Compute uncertainty by {uncert_metric}!")
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

#? self.uncert_all_info[single_idx][uncert_name] = 1 - cert_value
        
        # print("start comput_uncert process")
        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        # print("Done comput_uncert process")

        #todo augmented sample uncertainty is stored in self.uncert_all_info
        #todo no need to use sample actually must not!
        # print("uncert_all_info")
        # print(self.uncert_all_info)
        # print()
        # print(self.uncert_all_info.keys())
        # print("start variance_ratio process")
        for s_key in self.uncert_all_info.keys():
            self.variance_ratio(s_key, n_transforms)
        # print("Done variance_ratio process")

    # def variance_ratio(self, sample, cand_length):
        
    #     vote_counter = torch.zeros(sample["uncert_0"].size(0))
    #     for i in range(cand_length):
    #         top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
    #         vote_counter[top_class] += 1
    #     assert vote_counter.sum() == cand_length
    #     sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
    def variance_ratio(self, s_key, cand_length):
        # print(s_key)
        # print(self.uncert_all_info[s_key].keys())
        vote_counter = torch.zeros(self.uncert_all_info[s_key]["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(self.uncert_all_info[s_key][f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        self.uncert_all_info[s_key]["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
        # print(self.uncert_all_info[s_key].keys())
        # return sample

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
    
    
    def memory_config(self):
        memory_sampler = MemoryOrderedSampler(self.memory, self.batchsize, 1)
        memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=memory_sampler, num_workers=0, pin_memory=True)
        
        cnt = torch.zeros_like(torch.tensor(self.exposed_classes))
        print('Memory_sampler==>',len(self.memory))
        for idx, (_,target) in enumerate(memory_dataloader):
            print('memory idx',idx)
            for j in range(len(target)):
                target[j] = self.exposed_classes.index(target[j].item())
            cnt[target] +=1
            
    
        print("MEMORY BUFFER STATUS")
        print(cnt)
        