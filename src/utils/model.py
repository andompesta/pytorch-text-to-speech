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


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
