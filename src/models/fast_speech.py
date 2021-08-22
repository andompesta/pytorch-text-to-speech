from collections import OrderedDict
import os
import json
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.transformers import Encoder, Decoder, PostNet
from src.modules import VarianceAdaptor
from src.utils import (
    get_mask_from_lengths,
    fast_speech_weight_mapping
)


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(
        self,
        preprocess_config: dict,
        model_config: dict,
    ):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config        
        self.encoder = Encoder(model_config)

        self.variance_adaptor = VarianceAdaptor(
            preprocess_config,
            model_config
        )

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        
    def forward(
        self,
        speakers: torch.Tensor,
        phonems: torch.Tensor,
        phonems_lens: torch.Tensor,
        pitch_control: float,
        energy_control: float,
        duration_control: float
    ):
        max_phonems_len = phonems_lens.max().item()
        phonem_masks = get_mask_from_lengths(phonems_lens, max_phonems_len)

        output = self.encoder(phonems, phonem_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_phonems_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            phonem_masks,
            pitch_control=pitch_control,
            energy_control=energy_control,
            duration_control=duration_control
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            phonem_masks,
            mel_masks,
            phonems_lens,
            mel_lens,
        )

    @torch.jit.unused
    def to(
        self, 
        device: str
    ):
        self.variance_adaptor.length_regulator.device = device
        return super().to(device)

    @classmethod
    def build(
        cls, 
        preprocess_config: dict,
        model_config: dict,
        device: Union[torch.device, str] = "cpu",
        restore_step: Optional[int] = 900000,
        ckpt_path: Optional[str] = "./output/ckpt/LJSpeech",
        mapping_fn = fast_speech_weight_mapping,
        train: bool = False
    ):

        model = cls(
            preprocess_config,
            model_config,
        )
        if restore_step is not None and ckpt_path is not None:
            ckpt_path = os.path.join(
                ckpt_path,
                "{}.pth.tar".format(restore_step),
            )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model")
            strinct = True
            if mapping_fn is not None:
                state_dict = mapping_fn(state_dict)
                strict = False
            model.load_state_dict(state_dict, strict=strict)

        model = model.to(device)

        if train:
            model = model.train()
            model.requires_grad_ = True
        else:
            model = model.eval()
            model.requires_grad_ = False
        
        return model