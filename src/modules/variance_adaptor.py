from typing import Optional, Tuple, Union
import torch
import numpy as np
import json
import os

from torch import nn

from src.utils.tools import get_mask_from_lengths
from .variance_predictor import VariancePredictor
from .length_regulator import LengthRegulator


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(
        self,
        preprocess_config: dict,
        model_config: dict,
        device: str = "cpu"
    ):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator(device)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        mask: torch.Tensor,
        control: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        mask: torch.Tensor,
        control: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        pitch_control: float,
        energy_control: float,
        duration_control: float
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, 
            None,
            src_mask,
            pitch_control
        )
        x = x + pitch_embedding

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x,
            None,
            src_mask,
            energy_control
        )
        x = x + energy_embedding

        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * duration_control),
            min=0,
        )
        x, mel_len = self.length_regulator(
            x,
            duration_rounded,
        )
        max_mel_len = mel_len.max().item()
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )