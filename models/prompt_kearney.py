
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self,
                 pool_size=10,
                 selection_size=5,
                 prompt_len=5,
                 dimension=768,
                 _diversed_selection = True,
                 _batchwise_selection= False,
                 **kwargs):
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimension      = dimension
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimension, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimension, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)
    def forward(self, query : torch.Tensor,  **kwargs):

        B, D = query.shape
        assert D == self.dimension, f'Query dimension {D} does not Cdist prompt dimension {self.dimension}'
        c_dist= 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        _ ,topk = c_dist.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimension))
        distance = c_dist.gather(1, topk)

        return distance.mean(), selection