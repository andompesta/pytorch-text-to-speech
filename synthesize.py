import argparse
from typing import List, Tuple
import re
import yaml
import numpy as np
import torch
import os
from string import punctuation
from g2p_en import G2p
from pypinyin import pinyin, Style
from scipy.io import wavfile


from src.models import FastSpeech2, VocoderGenerator, Synthesizer
from src.utils import (
    to_device,
    Batch,
    vocoder_infer,
    pad_1D
)
from src.text import text_to_sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default="config/LJSpeech/preprocess.yaml"
    )
    parser.add_argument(
        "-m", 
        "--mel_config", 
        type=str, 
        default="config/LJSpeech/model.yaml" 
    )
    parser.add_argument(
        "-v", 
        "--voc_config", 
        type=str, 
        default="config/hifigan/model.yaml"
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default="config/LJSpeech/train.yaml"
    )

    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )

    return parser.parse_args()


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(
    texts: List[str],
    preprocess_config
) -> List[np.array]:
    sequences = []
    for text in texts:
        text = text.rstrip(punctuation)
        lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, 
                preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )

        sequences.append(sequence)
    return sequences

def synthesize(
    model: torch.nn.Module,
    configs: Tuple[dict, dict, dict],
    vocoder: torch.nn.Module,
    batchs: List[Batch],
    control_values: Tuple[float, float, float],
    device: str,
    output_path: str,
):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                batch.speakers,
                batch.phonems,
                batch.phonems_len,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            
            # generate wave
            mel_predictions = output[1].transpose(1, 2)
            lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
            wav_predictions = vocoder_infer(
                mel_predictions,
                vocoder,
                model_config,
                preprocess_config,
                lengths=lengths
            )

            # save melmogram
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            wavfile.write(
                os.path.join(
                    output_path, "{}.wav".format(batch.doc_id)
                ), 
                sampling_rate,
                np.concatenate(wav_predictions)
            )


if __name__ == "__main__":
    args = parse_args()
    device = "cpu"

    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)

    synthesizer = Synthesizer.build(
        args.preprocess_config,
        args.mel_config,
        args.voc_config,
        device
    ).eval()
    
    raw_texts = [
        "Learning feature interactions is crucial for click-through rate (CTR) prediction in recommender systems.",
        "In most existing deep learning models, feature interactions are either manually designed or simply enumerated.",
        "However, enumerating all feature interactions brings large memory and computation cost.",
        "Even worse, useless interactions may introduce noise and complicate the training process.",
        "In this work, we propose a two-stage algorithm called Automatic Feature Interaction Selection (AutoFIS).",
        "AutoFIS can automatically identify important feature interactions for factorization models with computational cost just equivalent to training the target model to convergence.",
        "In the search stage, instead of searching over a discrete set of candidate feature interactions, we relax the choices to be continuous by introducing the architecture parameters.",
        "By implementing a regularized optimizer over the architecture parameters, the model can automatically identify and remove the redundant feature interactions during the training process of the model.",
        "In the re-train stage, we keep the architecture parameters serving as an attention unit to further boost the performance.",
        "Offline experiments on three large-scale datasets (two public benchmarks, one private) demonstrate that AutoFIS can significantly improve various FM based models.",
        "AutoFIS has been deployed onto the training platform of Huawei App Store recommendation service, where a 10-day online A/B test demonstrated that AutoFIS improved the DeepFM model by 20.3% and 20.1% in terms of CTR and CVR respectively."
    ]

    speakers = np.array([args.speaker_id] * len(raw_texts))

    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        phonems = preprocess_english(
            raw_texts, 
            preprocess_config
        )
        
    else:
        NotImplementedError("{} language preprocessing not implemented".format(preprocess_config["preprocessing"]["text"]["language"]))
    
    phonems_len = np.array([len(p) for p in phonems])
    phonems = pad_1D(phonems)
    batch = Batch(
        doc_id="paper-4",
        texts=raw_texts, 
        speakers=speakers,
        phonems=phonems,
        phonems_len=phonems_len,
    )
    
    
    batch = to_device(batch, "cpu")

    example_inputs = (
        batch.speakers,
        batch.phonems,
        batch.phonems_len,
        1.0,
        1.0,
        1.0
    )

    with torch.no_grad():
        wavs, lengths = synthesizer(*example_inputs)


    gens = []
    for i, (wav, len) in enumerate(zip(wavs, lengths)):
        wav = wav[: len]
        gens.append(wav)
    

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(
        "output/result/LJSpeech/{}.wav".format(batch.doc_id), 
        sampling_rate,
        np.concatenate(gens)
    )

    print("done")
