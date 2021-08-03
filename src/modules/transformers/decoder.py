from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn

from src.text.symbols import symbols
from .fft_block import FFTBlock
from .utils import get_sinusoid_encoding_table

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, 
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        enc_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        max_len = min(max_len, self.max_seq_len)

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        dec_output = enc_seq[:, :max_len, :] + self.position_enc[
            :, :max_len, :
        ].expand(batch_size, -1, -1)
        mask = mask[:, :max_len]
        slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            
        return dec_output, mask