from typing import Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiHeadAttention
from .positionwise_feed_forward import PositionwiseFeedForward


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_inner: int,
        kernel_size: Tuple[int, int],
        dropout=0.1
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, 
            d_inner,
            kernel_size,
            dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn
