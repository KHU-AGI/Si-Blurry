from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import logging
import copy
import time
import datetime

import gc
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods._trainer import _Trainer

from utils.train_utils import select_optimizer, select_scheduler

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from models.vit import _create_vision_transformer
from utils.memory import MemoryBatchSampler
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

class Ours(_Trainer):
    def __init__(self, **kwargs):
        super(Ours, self).__init__(**kwargs)
        
        self.mask_viz = torch.empty(0)
        self.use_mask    = kwargs.get("use_mask")
        self.use_contrastiv  = kwargs.get("use_contrastiv")
        self.use_last_layer  = kwargs.get("use_last_layer")
        self.use_afs  = kwargs.get("use_afs")
        self.use_mcr  = kwargs.get("use_mcr")
        
        self.alpha  = kwargs.get("alpha")
        self.gamma  = kwargs.get("gamma")
        self.margin  = kwargs.get("margin")

        self.labels = torch.empty(0)
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        # for j in range(len(labels)):
        #     labels[j] = self.exposed_classes.index(labels[j].item())

        self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=4)
        self.memory_provider     = iter(self.memory_dataloader)

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels)
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        self.labels = torch.cat((self.labels, y), 0)
        
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in range(len(memory_labels)):
                memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)
        
        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x, y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if self.use_mask:
                logit = logit * mask
            logit = logit + self.mask
            loss = self.loss_fn(feature, mask, y)
        return logit, loss

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
                logit = logit + self.mask
                loss = F.cross_entropy(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()
            
    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        self.model_without_ddp.keys = torch.cat([self.model_without_ddp.keys, self.model_without_ddp.key.detach().cpu()], dim=0)
        self.mask_viz = torch.cat([self.mask_viz, self.model_without_ddp.mask.detach().cpu()], dim=0)
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.fc)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask
        
        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim
        
        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim
        
        return sample_grad, batch_grad
    
    def _get_ignore(self, sample_grad, batch_grad):
        # ign_score = torch.max(1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1), torch.zeros(1, device=self.device)) #B
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.model_without_ddp.backbone.fc.weight[y].clone().detach()
        # cps_score = torch.max(1 - torch.cosine_similarity(head_w, sample_g, dim=1), torch.ones(1, device=self.device)) # B
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score
    
    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(log_p, y)
        # mask_loss = F.cross_entropy(logit, y)

        # if self.use_afs:
        #     logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        # else:
        #     logit = self.model_without_ddp.forward_head(feature)
        # if self.use_mask:
        #     logit = logit * mask
        # logit = logit + self.mask
        # log_p = F.log_softmax(logit, dim=1)
        # loss = F.nll_loss(log_p, y, reduction='none')
        # loss = F.cross_entropy(logit, y, reduction='none')
        if self.use_mcr:
            loss = (1-self.alpha)* loss + self.alpha * (ign_score ** self.gamma) * loss
        # return loss.mean() + self.model_without_ddp.get_similarity_loss()
        return loss.mean() + self.model_without_ddp.get_similarity_loss()
    
    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
            f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
            f"N_Exposed {len(self.exposed_classes)} | "
            f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
        )

    def setup_distributed_model(self):
        super().setup_distributed_model()
        self.model_without_ddp.use_mask = self.use_mask
        self.model_without_ddp.use_contrastiv = self.use_contrastiv
        self.model_without_ddp.use_last_layer = self.use_last_layer


    # def main_worker(self, gpu) -> None:
    #     super(Ours, self).main_worker(gpu)

    #     vis_sel = torch.randperm(self.model_without_ddp.features.shape[0])[:10000]
    #     self.model_without_ddp.features = torch.cat([self.model_without_ddp.features[vis_sel], self.model_without_ddp.keys], dim=0)
    #     self.model_without_ddp.features = F.normalize(self.model_without_ddp.features, dim=1)

    #     tsne = TSNE(n_components=2, random_state=0)
    #     X_2d = tsne.fit_transform(self.model_without_ddp.features.detach().cpu().numpy())

    #     for t in range(5):
    #         for m in range(10):
    #             for i in range(100):
    #                 if torch.sigmoid(self.mask_viz[t*10+m][i]) > 0.5:
    #                     plt.scatter(X_2d[:10000][self.labels[vis_sel]==i, 0], X_2d[:10000][self.labels[vis_sel]==i, 1], s = 1, alpha=1)
    #                 else:
    #                     plt.scatter(X_2d[:10000][self.labels[vis_sel]==i, 0], X_2d[:10000][self.labels[vis_sel]==i, 1], s = 1, alpha=0.2)
    #             plt.scatter(X_2d[-50:-40, 0], X_2d[-50:-40, 1], s = 50, marker='^', c='black')
    #             for i in range(10):
    #                 plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
    #             plt.savefig(f'OURS_tsne{self.rnd_seed}_Task{t+1}_mask{m}.png')
    #             plt.clf()
                    
        # torch.save(self.mask_viz, f"mask_viz_{self.rnd_seed}.pth")
        # idx = torch.randperm(self.model_without_ddp.features.shape[0])

        # print(self.labels.size())
        # print(self.model_without_ddp.features.shape)
        # labels = self.labels[idx[:10000]]

        # self.model_without_ddp.features = torch.cat([self.model_without_ddp.features[idx[:10000]], self.model_without_ddp.keys], dim=0)
        # self.model_without_ddp.features = F.normalize(self.model_without_ddp.features, dim=1)

        # tsne = TSNE(n_components=2, random_state=0)
        # X_2d = tsne.fit_transform(self.model_without_ddp.features.detach().cpu().numpy())
        
        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-50:-40, 0], X_2d[-50:-40, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'OURS_tsne{self.rnd_seed}_Task1.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-40:-30, 0], X_2d[-40:-30, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'OURS_tsne{self.rnd_seed}_Task2.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-30:-20, 0], X_2d[-30:-20:, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'OURS_tsne{self.rnd_seed}_Task3.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-20:-10, 0], X_2d[-20:-10, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'OURS_tsne{self.rnd_seed}_Task4.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-10:, 0], X_2d[-10:, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'OURS_tsne{self.rnd_seed}_Task5.png')
        # plt.clf()
       
    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,)).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier() # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]], index)
            else:
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])