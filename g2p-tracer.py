import os

from dp.phonemizer import Phonemizer

if __name__ == "__main__":
    phonemizer = Phonemizer.from_checkpoint("checkpoints/g2p/en_us_cmudict_forward.pt")

    result = phonemizer.phonemise_list(
        ["Phonemizing an English text is imposimpable!"], lang="en_us"
    )

    for word, pred in result.predictions.items():
        print(f"{word} {pred.phonemes} {pred.confidence}")

    with open(os.path.join("lexicon", "phonemizer.txt"), "w") as writer:
        for w, p in phonemizer.lang_phoneme_dict["en_us"].items():
            writer.write("{}\t{}\n".format(w, p))

    # model = phonemizer.predictor.model
    # traced_model = torch.jit.script(model)

    # print(traced_model.graph)

    # traced_model.save(
    #     os.path.join(
    #         "checkpoints/traces/"
    #         "g2p-traced.pt"
    #     )
    # )

    output = phonemizer("phonemise this with a traced model", "en_us")
    print(output)
