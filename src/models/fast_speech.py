from collections import OrderedDict
import os
import json
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.transformers import Encoder, Decoder, PostNet
from src.modules import VarianceAdaptor
from src.utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(
        self,
        preprocess_config: dict,
        model_config: dict
    ):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
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
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
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
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
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
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    @classmethod
    def build(
        cls, 
        preprocess_config: dict,
        model_config: dict,
        restore_step: Optional[int] = 900000,
        ckpt_path: Optional[str] = "./output/ckpt/LJSpeech",
        device: Union[torch.device, str] = "cpu",
        train: bool = False
    ):

        model = cls(preprocess_config, model_config)
        if restore_step is not None and ckpt_path is not None:
            ckpt_path = os.path.join(
                ckpt_path,
                "{}.pth.tar".format(restore_step),
            )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_model = OrderedDict()
            for name, param in ckpt["model"].items():
                if "conv_layer" is in name:
                    name.split("conv")                
                else:
                    ckpt_model[name] = param
            model.load_state_dict(ckpt["model"])

        model = model.to(device)

        if train:
            model = model.train()
            model.requires_grad_ = True
        else:
            model = model.eval()
            model.requires_grad_ = False
        
        return model