"""Small script to preprocess the raw data that is generated and make it suitable for the network

"""

from languageDetectinator.datasets import Vocabulary
import numpy as np

def outputVector(index: int, totalLangs: int) -> np.array:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return np.array([float(v) for v in stringVec])

def main():
    languages = ["en","fr","de","es","cs","pt","pl","ar","ru","ja"]

    vecs = []
    for i,l in enumerate(languages):
        with open(f"./data/raw/{l}_text.txt","r") as file:
            vocab = Vocabulary(file.read())

        vocab.pruneVocabulary(10)
        vocabVec = vocab.vectorizeVocabulary(10)
        outVec = outputVector(i,10)
        modelVec = [outVec,vocabVec]

        vecs.append(modelVec)

if __name__ == "__main__":
    main()