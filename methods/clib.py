import logging
import copy
import time
import math

import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import ttest_ind

import gc
from methods.er_baseline import ER
from utils.memory import MemoryBatchSampler, MemoryOrderedSampler
from utils.memory import Memory
from datasets import *
from utils.onlinesampler import OnlineSampler, OnlineTestSampler

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLIB(ER):
    def __init__(self, *args, **kwargs):
        super(CLIB, self).__init__(*args, **kwargs)
        self.loss = torch.empty((0,))
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.imp_update_period = kwargs['imp_update_period']
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'

        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr

    def setup_distributed_dataset(self):
        super(CLIB, self).setup_distributed_dataset()
        # _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        # _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        # self.train_dataset   = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, 
        #                                               transform=self.train_transform)
        # self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        # self.train_sampler   = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        # self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler, num_workers=self.n_worker, pin_memory=True)
        self.loss_update_dataset = self.datasets[self.dataset](root=self.data_dir, train=True, download=True,
                                     transform=transforms.Compose([transforms.Resize((self.inp_size,self.inp_size)),
                                                                   transforms.ToTensor()]))
        self.memory = Memory(data_source=self.loss_update_dataset)

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self.update_memory(idx, labels)
        self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=0)
        self.memory_provider     = iter(self.memory_dataloader)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter) * self.temp_batchsize * self.world_size): # * self.temp_batchsize * self.world_size
            loss, acc = self.online_train([torch.empty(0), torch.empty(0)])
            _loss += loss
            _acc += acc
            _iter += 1
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

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
                score = self.memory.others_loss_decrease[cand_idx]
                idx_to_replace = cand_idx[torch.argmin(score)]
                self.memory.replace_data([x, y], idx_to_replace)
                self.dropped_idx.append(idx_to_replace)
                self.memory_dropped_idx.append(idx_to_replace)
            else:
                self.memory.replace_data([x, y])
                self.dropped_idx.append(len(self.memory) - 1)
                self.memory_dropped_idx.append(len(self.memory) - 1)

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.samplewise_loss_update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)


        return total_loss, total_correct / total_num_data

    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            if len(self.loss) == 0: return
            mask = torch.ones(len(self.loss), dtype=bool)
            mask[torch.tensor(self.dropped_idx, dtype=torch.int64).squeeze()] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(torch.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]).cpu())
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(torch.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]).cpu())
                        if len(self.low_lr_loss) > min_iter:
                            del self.low_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = True
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.high_lr
                        param_group["initial_lr"] = self.high_lr
                self.dropped_idx = []
                if len(self.high_lr_loss) == len(self.low_lr_loss) and len(self.high_lr_loss) >= min_iter:
                    stat, pvalue = ttest_ind(self.low_lr_loss, self.high_lr_loss, equal_var=False, alternative='greater')
                    # print(pvalue)
                    if pvalue < significance:
                        self.high_lr = self.low_lr
                        self.low_lr *= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
                    elif pvalue > 1 - significance:
                        self.low_lr = self.high_lr
                        self.high_lr /= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    logit = []
                    label = []
                    for (x, y) in self.loss_update_dataloader:
                        logit.append(self.model(x.to(self.device)) + self.mask)
                        label.append(y.to(self.device))
                    loss = F.cross_entropy(torch.cat(logit), torch.cat(label), reduction='none')
                    if self.distributed:
                        loss = torch.cat(self.all_gather(loss), dim=-1).flatten()
                    loss = loss.cpu()
                    self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                    self.memory_dropped_idx = []
                self.loss = loss

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    logit = []
                    label = []
                    # loss = []
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        for i in range(0, math.ceil(len(self.memory) / batchsize)):
                            logit.append(self.model(torch.cat(self.memory.images[i*batchsize:min((i+1)*batchsize, len(self.memory)):self.world_size]).to(self.device)) + self.mask)
                            label.append(self.memory.labels[i*batchsize:min((i+1)*batchsize, len(self.memory)):self.world_size].to(self.device))
                        logits = torch.cat(logit)
                        labels = torch.cat(label)
                    #         logit = self.model(torch.cat(self.memory.images[i*batchsize:min((i+1)*batchsize, len(self.memory)):self.world_size]).to(self.device)) + self.mask
                    #         label = self.memory.labels[i*batchsize:min((i+1)*batchsize, len(self.memory)):self.world_size].to(self.device)
                    #         loss.append(F.cross_entropy(logit, label.to(torch.int64), reduction='none'))
                    # loss = torch.cat(loss)
                        loss = F.cross_entropy(logits, labels.to(torch.int64), reduction='none')
                        if self.distributed:
                            loss = torch.cat(self.all_gather(loss), dim=-1).flatten()
                    loss = loss.cpu()
                    self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                    self.memory_dropped_idx = []
                self.loss = loss
