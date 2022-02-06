import argparse
import re
from string import punctuation
from typing import Dict, List

import nltk
import numpy as np
import torch
import yaml
from g2p_en import G2p
from scipy.io import wavfile

from src.inference.arxiv_api import get_arxiv_articles
from src.inference.scholar_api import get_authors_citations
from src.text import text_to_sequence
from src.utils import Batch, pad_1D, to_device


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
        default="config/LJSpeech/preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--mel_config",
        type=str,
        default="config/LJSpeech/model.yaml",
    )
    parser.add_argument(
        "-v",
        "--voc_config",
        type=str,
        default="config/hifigan/model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default="config/LJSpeech/train.yaml",
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


def read_lexicon(lex_path) -> Dict[str, str]:
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
    texts: List[str], preprocess_config, lexicon: Dict[str, str]
) -> List[np.array]:
    sequences = []
    for text in texts:
        text = text.rstrip(punctuation)

        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        words = filter(lambda x: x != " ", words)

        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += g2p(w)
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones,
                preprocess_config["preprocessing"]["text"]["text_cleaners"],
            )
        )
        sequences.append(sequence)
    return sequences


if __name__ == "__main__":
    args = parse_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"),
        Loader=yaml.FullLoader,
    )

    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    sent_detector = nltk.data.load(
        "/home/ando/nltk_data/tokenizers/punkt/english.pickle"
    )
    synthesizer = torch.jit.load("./checkpoints/synthesizer/traced.pt")

    articels = list(get_arxiv_articles())
    for r in articels:
        citations = 0
        for a in r.authors:
            authors_data = get_authors_citations(a.name)
            citation = authors_data.get("citation")
            citations += citation
        r.score = citations / len(r.authors)

    articels = sorted(articels, key=lambda r: r.score, reverse=True)

    vocalaized_summaries = []

    for idx, a in enumerate(articels):

        title_sentences = sent_detector.tokenize(a.title.strip())
        abstract_sentences = sent_detector.tokenize(a.summary.strip())

        abstract_sentences = [s.replace("e.g.", "for example").replace("i.e.", "that is") for s in abstract_sentences]

        if title_sentences[-1][-1] != ".":
            title_sentences[-1] = title_sentences[-1] + "."

        if idx == 0:
            content = [
                "Welcome to Zalando tech paper review.",
                f"Paper number: {idx + 1}.",
            ]
        else:
            content = [
                ".",
                f"Paper number: {idx + 1}.",
            ]

        content += title_sentences

        content += ["\n."]
        content += abstract_sentences

        speakers = np.array([args.speaker_id] * len(content))
        phonems = preprocess_english(content, preprocess_config, lexicon=lexicon)

        phonems_len = np.array([len(p) for p in phonems])
        phonems = pad_1D(phonems)
        batch = Batch(
            doc_id="tracing",
            texts=content,
            speakers=speakers,
            phonems=phonems,
            phonems_len=phonems_len,
        )

        batch = to_device(batch, "cpu")

        inputs = dict(
            speakers=batch.speakers,
            phonems=batch.phonems,
            phonems_len=batch.phonems_len,
            pitch_control=args.pitch_control,
            energy_control=args.energy_control,
            duration_control=args.duration_control,
        )

        with torch.no_grad():
            wavs, lengths = synthesizer(**inputs)

        gens = []
        for i, (wav, length) in enumerate(zip(wavs, lengths)):
            wav = wav[:length]
            gens.append(wav)

        vocalaized_summaries.append(
            np.concatenate(gens),
        )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(
        "{}.wav".format(batch.doc_id),
        sampling_rate,
        np.concatenate(vocalaized_summaries),
    )

    print("done")
