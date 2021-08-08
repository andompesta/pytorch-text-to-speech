from typing import List
from torch import nn, Tensor, jit
from torch.nn import functional as F
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from .residual import ResBlock
from .utils import init_weights, init_weights, LRELU_SLOPE



class UpSampler(nn.Module):
    def __init__(
        self,
        upsample_in_channel: int,
        upsample_out_channel: int,
        upsample_kernel_size: int,
        upsample_rates: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]]
    ):
        super(UpSampler, self).__init__()

        self.up = weight_norm(
            ConvTranspose1d(
                upsample_in_channel,
                upsample_out_channel,
                upsample_kernel_size,
                upsample_rates,
                padding=(upsample_kernel_size - upsample_rates) // 2,
            )
        )
        self.up.apply(init_weights)

        self.res_0 = ResBlock(
            channels=upsample_out_channel,
            kernel_size=resblock_kernel_sizes[0],
            dilation=resblock_dilation_sizes[0]
        )

        self.res_1 = ResBlock(
            channels=upsample_out_channel,
            kernel_size=resblock_kernel_sizes[1],
            dilation=resblock_dilation_sizes[1]
        )

        self.res_2 = ResBlock(
            channels=upsample_out_channel,
            kernel_size=resblock_kernel_sizes[2],
            dilation=resblock_dilation_sizes[2]
        )
        self.num_kernels = len(resblock_kernel_sizes)

    def forward(
        self,
        x: Tensor,
        lrelu_slope : float = LRELU_SLOPE
    ) -> Tensor:
        x = F.leaky_relu(x, lrelu_slope)
        x = self.up(x)

        xs = self.res_0(x)
        xs += self.res_1(x)
        xs += self.res_2(x)

        x = xs / self.num_kernels
        
        return x

    @jit.ignore
    def remove_wn(self):
        remove_weight_norm(self.up)
        self.res_0.remove_wn()
        self.res_1.remove_wn()
        self.res_2.remove_wn()
        