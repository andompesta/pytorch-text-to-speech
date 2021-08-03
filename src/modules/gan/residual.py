from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1

def init_weights(
    m: nn.Module,
    mean=0.0,
    std=0.01
):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(
    kernel_size: int,
    dilation=1
) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    def __init__(
        self, 
        config: dict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5)
    ):
        super(ResBlock, self).__init__()
        self.config = config
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(
        self,
        x: Tensor,
        lrelu_slope : float = LRELU_SLOPE
    ) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)