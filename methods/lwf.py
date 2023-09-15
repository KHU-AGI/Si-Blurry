# When we make a new one, we should inherit the Finetune class.
import gc
import copy
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
from utils.train_utils import select_scheduler
from methods.er_baseline import ER
from utils.memory import Memory

logger = logging.getLogger()

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class LwF(ER):
    def __init__(self, *args, **kwargs):
        super(LwF, self).__init__(*args, **kwargs)
        # self.prev_fc=None
        self.kd_hp =0.2
        self.task_id=None
        self.old_model =None
        self.old_mask = None
        
    def setup_distributed_dataset(self):
        super(LwF, self).setup_distributed_dataset()
        self.loss_update_dataset = self.datasets[self.dataset](root=self.data_dir, train=True, download=True,
                                     transform=transforms.Compose([transforms.Resize((self.inp_size,self.inp_size)),
                                                                   transforms.ToTensor()]))
        self.memory = Memory(data_source=self.loss_update_dataset)
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        # self.update_memory(idx, labels)
        self.old_model = self.freeze(copy.deepcopy(self.model))
        self.old_mask = copy.deepcopy(self.mask)
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def add_new_class(self, class_name):
        # For DDP, normally go into this function
        len_class = len(self.exposed_classes)
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
    
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
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]], index)
            else:
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])
    
    def online_before_task(self, task_id):
        pass
    
    def freeze(self,model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        return model
    
    def online_after_task(self,task_id):
        pass
        
    def _KD_loss(self,pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data

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
    
    def model_forward(self, x, y):
        kd_loss =0.
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            ori_logit = self.model(x)
            logit = ori_logit+self.mask
            loss = self.criterion(logit, y)
            if self.old_model is not None:
                old_logit = self.old_model(x)
                kd_loss = self._KD_loss(ori_logit[:,:len(self.old_mask)],
                                        old_logit[:,:len(self.old_mask)],T=2.)
                loss += self.kd_hp * kd_loss
            
        # loss = ce_loss + self.kd_hp * kd_loss
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
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
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