import argparse
from typing import List
import re
import yaml
import numpy as np
import torch
from string import punctuation
from g2p_en import G2p


from src.models import Synthesizer, synthesizer
from src.utils import (
    to_device,
    Batch,
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
        "lets try to trace this",
        "maybe will works eventually"
    ]

    speakers = np.array([args.speaker_id] * len(raw_texts))

    phonems = preprocess_english(
        raw_texts, 
        preprocess_config
    )
        
    phonems_len = np.array([len(p) for p in phonems])
    phonems = pad_1D(phonems)
    batch = Batch(
        doc_id="tracing",
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

    # control input
    control_raw_texts = [
        "lets try to trace this out",
    ]

    control_speakers = np.array([args.speaker_id] * len(control_raw_texts))

    control_phonems = preprocess_english(
        control_raw_texts, 
        preprocess_config
    )
        
    control_phonems_len = np.array([len(p) for p in control_phonems])
    control_phonems = pad_1D(control_phonems)
    control_batch = Batch(
        doc_id="tracing",
        texts=control_raw_texts, 
        speakers=control_speakers,
        phonems=control_phonems,
        phonems_len=control_phonems_len,
    )
    
    
    control_values = args.pitch_control, args.energy_control, args.duration_control

    control_batch = to_device(control_batch, device)

    control_example_inputs = (
        control_batch.speakers,
        control_batch.phonems,
        control_batch.phonems_len,
        1.0,
        1.0,
        1.0
    )
    
    jit_module = torch.jit.script(synthesizer)

    print(jit_module.graph)

    jit_module(*example_inputs)
    jit_module(*control_example_inputs)

    jit_module.save("traced.pt")