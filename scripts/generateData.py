"""Quick script to generate some basic data for us to start training some models with!

"""

from languageDetectinator.datasets import Vocabulary
import requests
import json
import numpy as np
from unidecode import unidecode

def getLanguageWords(jsonFile: str) -> Vocabulary:
    response = requests.get(jsonFile)
    words = ""

    for line in response.iter_lines():
        j = json.loads(line)
        words += f"{unidecode(j['word'])} "

    return Vocabulary(words)

def outputVector(index: int, totalLangs: int) -> np.array:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return np.array([float(v) for v in stringVec])

def main():
    with open("data/raw/languageDictionaries.txt", "r") as f:
        languageFiles = f.readlines()

    vecs = []
    for i, jsonFile in enumerate(languageFiles):
        jsonFile = jsonFile.strip()
        print(f"Getting language {i+1} of {len(languageFiles)} from: {jsonFile}")
        vocab = getLanguageWords(jsonFile)

        vocab.pruneVocabulary(12)
        print(f"Found {len(vocab.words)} unique words.")
        vocabVec = vocab.vectorizeVocabulary(12)
        outVec = outputVector(i, len(languageFiles)).reshape(1, len(languageFiles))

        combVec = np.concatenate((vocabVec,np.repeat(outVec,vocabVec.shape[0],axis=0)),axis=1)
        vecs.append(combVec)

    vecs = np.vstack(vecs)
    np.save("./data/processed/vectors.npy", vecs)

if __name__ == "__main__":
    main()