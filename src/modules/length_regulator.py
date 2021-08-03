from typing import List, Tuple
import torch

from src.utils.tools import pad

class LengthRegulator(torch.nn.Module):
    """ Length Regulator """

    def __init__(
        self,
        device: str
    ):
        super(LengthRegulator, self).__init__()
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        output = torch.jit.annotate(List[torch.Tensor], [])
        mel_len = torch.jit.annotate(List[int], [])

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        output = pad(output)

        return output, torch.tensor(mel_len).long().to(self.device)

    def expand(
        self,
        batch: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out