from typing import List, Optional, Union
import torch
from torch import nn


from src.utils.tools import pad


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(
        self,
    ):
        super(LengthRegulator, self).__init__()
        
    def LR(
        self,
        x: List[List[torch.Tensor]],
        duration: List[torch.Tensor],
        max_len: Optional[int],
        device: Union[torch.device, str]
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

        return output, torch.LongTensor(mel_len).to(device)

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

    def forward(
        self,
        x,
        duration,
        max_len,
        device    
    ):
        output, mel_len = self.LR(
            x,
            duration,
            max_len,
            device
        )
        return output, mel_len