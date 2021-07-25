from collections import OrderedDict
import os
import json
from typing import Tuple

import torch
import numpy as np



def fast_speech_weight_mapping(
    state_dcit: OrderedDict
) -> OrderedDict:
    ckpt_model = OrderedDict()
    for name, param in state_dcit.items():
        if "conv." in name:
            tokens = name.split("conv.")
            ckpt_model[tokens[0] + tokens[1]] = param
        else:
            ckpt_model[name] = param
    return ckpt_model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

