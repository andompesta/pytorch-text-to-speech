import os
import json
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np


def get_mask_from_lengths(
    lengths,
    max_len: Optional[int]=None,
    device: str = "cpu"
):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def pad_1D(
    inputs: List[np.array],
    PAD=0
) -> np.array:
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

torch.jit.script
def pad(
    input_ele: List[torch.Tensor],
) -> torch.Tensor:
    max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])
    output = torch.jit.annotate(List[torch.Tensor], [])

    for i, batch in enumerate(input_ele):
        one_batch_padded = F.pad(
            batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
        )
        output.append(one_batch_padded)
    output = torch.stack(output)
    return output
