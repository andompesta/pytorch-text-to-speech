from typing import Tuple
from torch import nn, Tensor
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(
        self,
        d_in: int, 
        d_hid: int,
        kernel_size: Tuple[int, int], 
        dropout=0.1
    ):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: Tensor
    ):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output