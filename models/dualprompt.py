from typing import TypeVar, Iterable
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs,_create_vision_transformer

logger = logging.getLogger()

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
    return model

class Prompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = False,
                 _batchwise_selection : bool = False,
                 **kwargs):
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor, **kwargs):
        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
        # Select prompts
        match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        # match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Frequency counter
        self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
        simmilarity = match.gather(1, topk)
        # get unsimilar prompts also 
        return simmilarity, selection

    def update(self):
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        else:
            return counter

class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = 'vit_base_patch16_224_l2p',
                 **kwargs):
        super().__init__()

        # self.features = torch.empty(0)
        # self.keys     = torch.empty(0)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(num_classes))
        
        self.lambd      = lambd
        self.class_num  = num_classes

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.tasks = []

        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        g_pool = 1
        e_pool = task_num
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        
        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        
        self.task_num = task_num
        self.task_id = -1 # if _convert_train_task is not called, task will undefined

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x
    
    
    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1]), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x

    def forward(self, inputs : torch.Tensor) :
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0].expand(B, -1, -1)
        else:
            g_p = None
        if self.e_prompt is not None:
                e_s, e_p = self.e_prompt(query)
        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        x = self.backbone.head(x[:, 0])

        self.similarity = e_s.mean()
        return x

    def convert_train_task(self, task : torch.Tensor, **kwargs):
    
        task = torch.tensor(task,dtype=torch.float)
        flag = -1
        for n, t in enumerate(self.tasks):
            if torch.equal(t, task):
                flag = n
                break
        if flag == -1:
            self.tasks.append(task)
            self.task_id = len(self.tasks) - 1
            if self.training:
                if self.task_id != 0:
                    with torch.no_grad():
                        self.e_prompt.prompts[self.task_id] = self.e_prompt.prompts[self.task_id - 1].clone()
                        self.e_prompt.key[self.task_id] = self.e_prompt.key[self.task_id - 1].clone()
        else :
            self.task_id = flag
        return
        
    def get_count(self):
        return self.e_prompt.update()

    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.similarity