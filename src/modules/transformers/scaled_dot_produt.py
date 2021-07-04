from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(
        self, 
        temperature: float
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn