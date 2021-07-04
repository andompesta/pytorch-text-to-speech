from typing import List, Optional, Union
import torch
from torch import nn


from src.utils.tools import pad


class LengthRegulator(nn.Module):
    """ Length Regulator """

    @property
    def device(self):
        return self._device

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None
    ):
        super(LengthRegulator, self).__init__()
        if device is not None:
            self._device = device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def LR(
        self,
        x: List[List[torch.Tensor]],
        duration: List[torch.Tensor],
        max_len: Optional[int]
    ):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(self.device)

    def expand(
        self,
        batch: List[torch.Tensor],
        predicted: torch.Tensor
    ) -> List[torch.Tensor]:
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len