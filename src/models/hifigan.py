from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from src.modules.gan import UpSampler, init_weights


class VocoderGenerator(nn.Module):
    def __init__(self, config: dict):
        super(VocoderGenerator, self).__init__()
        self.config = config
        self.num_kernels = len(config.get("resblock_kernel_sizes"))
        self.num_upsamples = len(config.get("upsample_rates"))
        self.conv_pre = weight_norm(
            Conv1d(80, config.get("upsample_initial_channel"), 7, 1, padding=3)
        )
        self.conv_pre.apply(init_weights)

        self.upsampler = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.get("upsample_rates"), config.get("upsample_kernel_sizes"))
        ):
            ch_in = config.get("upsample_initial_channel") // (2 ** i)
            ch_out = config.get("upsample_initial_channel") // (2 ** (i + 1))
            self.upsampler.append(
                UpSampler(
                    upsample_in_channel=ch_in,
                    upsample_out_channel=ch_out,
                    upsample_kernel_size=k,
                    upsample_rates=u,
                    resblock_kernel_sizes=config.get("resblock_kernel_sizes"),
                    resblock_dilation_sizes=config.get("resblock_dilation_sizes"),
                )
            )

        self.conv_post = weight_norm(Conv1d(ch_out, 1, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i, ups in enumerate(self.upsampler):
            x = ups(x)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = F.tanh(x)

        return x

    @torch.jit.ignore
    def remove_wn(self):
        print("Removing weight norm...")

        for ups in self.upsampler:
            ups.remove_wn()

        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def build(cls, config: dict, speaker: str = "LJSpeech", device: str = "cpu"):
        vocoder = cls(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(
                "./output/hifigan/generator_mapped_LJSpeech.pth.tar", map_location="cpu"
            )

        elif speaker == "universal":
            ckpt = torch.load(
                "./output/hifigan/generator_universal.pth.tar", map_location="cpu"
            )
        else:
            raise NotImplementedError(f"speaker {speaker} not implemented")

        vocoder.load_state_dict(ckpt)
        vocoder.eval()
        vocoder.remove_wn()
        return vocoder.to(device)


if __name__ == "__main__":
    import yaml

    ckpt = torch.load("./output/hifigan/generator_LJSpeech.pth.tar", map_location="cpu")

    vocoder_config = yaml.load(
        open("config/hifigan/model.yaml", "r"), Loader=yaml.FullLoader
    )
    vocoder = VocoderGenerator(vocoder_config)
    weights = list(vocoder.named_parameters())

    mapped_ckpt = OrderedDict()

    for name, p in ckpt["generator"].items():

        if name.startswith("ups"):
            _, level, weight = name.split(".")
            mapped_ckpt[f"upsampler.{level}.up.{weight}"] = p
        elif name.startswith("resblocks"):
            _, level, module, module_level, weight = name.split(".")
            res_level = int(level) % 3
            level = int(level) // 3
            mapped_ckpt[
                f"upsampler.{level}.res_{res_level}.{module}.{module_level}.{weight}"
            ] = p
        else:
            mapped_ckpt[name] = p

    vocoder.load_state_dict(mapped_ckpt)

    torch.save(
        vocoder.state_dict(), "./output/hifigan/generator_mapped_LJSpeech.pth.tar"
    )
