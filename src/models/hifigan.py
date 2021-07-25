import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.modules.gan import (
    ResBlock,
    init_weights,
    LRELU_SLOPE
)

class Generator(nn.Module):
    def __init__(
        self, 
        config: dict
    ):
        super(Generator, self).__init__()
        self.config = config
        self.num_kernels = len(config.get("resblock_kernel_sizes"))
        self.num_upsamples = len(config.get("upsample_rates"))
        self.conv_pre = weight_norm(
            Conv1d(80, config.get("upsample_initial_channel"), 7, 1, padding=3)
        )
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                config.get("upsample_rates"),
                config.get("upsample_kernel_sizes")
            )
        ):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        config.get("upsample_initial_channel") // (2 ** i),
                        config.get("upsample_initial_channel") // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.get("upsample_initial_channel") // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    config.get("resblock_kernel_sizes"),
                    config.get("resblock_dilation_sizes")
                )
            ):
                self.resblocks.append(resblock(config, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(
        self, 
        x
    ):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = F.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
    
    @classmethod
    def build(
        cls,
        config: dict,
        speaker: str = "LJSpeech",
        device: str = "cpu"
    ):
        vocoder = cls(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(
                "./output/hifigan/generator_LJSpeech.pth.tar", 
                map_location="cpu"
            )

        elif speaker == "universal":
            ckpt = torch.load(
                "./output/hifigan/generator_universal.pth.tar",
                map_location="cpu"    
            )
        else:
            raise NotImplementedError(f"speaker {speaker} not implemented")
        
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder.to(device)