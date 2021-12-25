import os
import re


if __name__ == "__main__":
    lexicon = dict()

    with open(os.path.join("./lexicon", "librispeech-lexicon.txt"), "r", encoding="utf8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones

    already_present = 0
    count = 0
    with open(os.path.join("./lexicon", "homograph-en.txt"), "r", encoding="utf8") as f:
        for line in f.read().splitlines():
            if line.startswith("#"):
                continue    # comment

            word, phones, pron2, pos1 = line.strip().split("|")
            phones = re.split(r"\s+", phones.strip("\n"))

            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
            else:
                already_present += 1
            count += 1
    
    print(already_present, count)
    print(len(lexicon))
    
    with open("lexicon.txt", "w") as f:
        for word, phones in lexicon.items():
            f.write("{}\t{}\n".format(
                word,
                " ".join(phones)
            ))
    