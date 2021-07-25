import argparse
from typing import List, Tuple
import re
import yaml
import numpy as np
import torch
from string import punctuation
from g2p_en import G2p
from pypinyin import pinyin, Style


from src.models import FastSpeech2, Generator
from src.utils import (
    synth_samples,
    to_device,
    pad_1D
)
from src.text import text_to_sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
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
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        required=True, 
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        required=True,
        help="path to train.yaml"
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

# def preprocess_english(
#     texts: List[str], 
#     preprocess_config: dict
# ) -> List[np.array]:
#     sequences = []
#     for t in texts:
#         text = t.rstrip(punctuation)
#         lexicon = read_lexicon(
#             preprocess_config["path"]["lexicon_path"]
#         )

#         g2p = G2p()
#         phones = []
#         words = re.split(r"([,;.\-\?\!\s+])", text)
#         for w in words:
#             if w.lower() in lexicon:
#                 phones += lexicon[w.lower()]
#             else:
#                 phones += list(filter(lambda p: p != " ", g2p(w)))
        
#         phones = "{" + "}{".join(phones) + "}"
#         phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
#         phones = phones.replace("}{", " ")

#         print("Raw Text Sequence: {}".format(text))
#         print("Phoneme Sequence: {}".format(phones))
#         sequence = np.array(
#             text_to_sequence(
#                 phones, 
#                 preprocess_config["preprocessing"]["text"]["text_cleaners"]
#             )
#         )
#         sequences.append(sequence)

#     return sequences

def preprocess_english(
    text, 
    preprocess_config
) -> np.array:
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
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def synthesize(
    model: torch.nn.Module, 
    configs: Tuple[dict, dict, dict],
    vocoder: torch.nn.Module,
    batchs: List[tuple],
    control_values: Tuple[float, float, float]
):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

if __name__ == "__main__":
    args = parse_args()
    device = "cpu"
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    
    model = FastSpeech2.build(
        preprocess_config,
        model_config,
        device=device
    )

    name = model_config["vocoder"]["model"]
    speaker = model_config["vocoder"]["speaker"]

    if name == "HiFi-GAN":
        vocoder_config = yaml.load(open("config/hifigan/model.yaml", "r"), Loader=yaml.FullLoader)
        vocoder = Generator.build(
            vocoder_config,
            speaker=speaker,
            device=device
        )
    else:
        raise NotImplementedError(name)
    
    ids = raw_texts = [args.text[:999]]

    # ids = [s for s in ids]
    # raw_texts = [s for s in raw_texts]
    # speakers = np.array([args.speaker_id] * len(ids))

    # if preprocess_config["preprocessing"]["text"]["language"] == "en":
    #     phonems = preprocess_english(
    #         ids,
    #         preprocess_config
    #     )
    # else:
    #     NotImplementedError("{} language preprocessing not implemented".format(preprocess_config["preprocessing"]["text"]["language"]))
    

    # phonem_lens = [p.shape[0] for p in phonems]
    # phonems = pad_1D(phonems)

    # batchs = [
    #     (ids, raw_texts, speakers, phonems, phonem_lens, max(phonem_lens))
    # ]
    
    ids = raw_texts = [args.text[:999]]
    speakers = np.array([args.speaker_id])

    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([
            preprocess_english(
                args.text, 
                preprocess_config
            )
        ])
    else:
        NotImplementedError("{} language preprocessing not implemented".format(preprocess_config["preprocessing"]["text"]["language"]))
    
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    configs = (preprocess_config, model_config, train_config)

    synthesize(model, configs, vocoder, batchs, control_values)
