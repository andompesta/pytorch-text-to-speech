import torch
import yaml

from .fast_speech import FastSpeech2
from .hifigan import VocoderGenerator


class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        preprocess_config: dict,
        mel_generator_config: dict,
        voice_generator_config: dict,
        device: str,
    ):
        super().__init__()

        self.preprocess_config = preprocess_config
        self.hop_length = self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.max_wav_value = self.preprocess_config["preprocessing"]["audio"][
            "max_wav_value"
        ]

        self.mel_generator_config = mel_generator_config
        self.voice_generator_config = voice_generator_config

        self.mel_generator = FastSpeech2.build(
            preprocess_config, mel_generator_config, device=device
        )

        self.vocoder_generator = VocoderGenerator.build(
            voice_generator_config, device=device
        )

    def forward(
        self,
        speakers: torch.Tensor,
        phonems: torch.Tensor,
        phonems_len: torch.Tensor,
        pitch_control: float,
        energy_control: float,
        duration_control: float,
    ):

        output = self.mel_generator(
            speakers,
            phonems,
            phonems_len,
            pitch_control=pitch_control,
            energy_control=energy_control,
            duration_control=duration_control,
        )

        # generate wave
        mel_predictions = output[1].transpose(1, 2)
        lengths = output[9] * self.hop_length

        wavs = self.vocoder_generator(mel_predictions).squeeze(1)

        wavs = (wavs.detach().cpu() * self.max_wav_value).short()

        return wavs, lengths

    @classmethod
    def build(
        cls,
        preprocess_config_path: str,
        mel_generator_config_path: str,
        voice_generator_config_path: str,
        device: str,
    ):

        preprocess_config = yaml.load(
            open(preprocess_config_path, "r"),
            Loader=yaml.FullLoader,
        )
        mel_generator_config = yaml.load(
            open(mel_generator_config_path, "r"),
            Loader=yaml.FullLoader,
        )
        voice_generator_config = yaml.load(
            open(voice_generator_config_path, "r"),
            Loader=yaml.FullLoader,
        )

        return cls(
            preprocess_config, mel_generator_config, voice_generator_config, device
        )
