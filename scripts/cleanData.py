"""Small script to preprocess the raw data that is generated and make it suitable for the network

"""

from languageDetectinator.datasets import Vocabulary
import numpy as np

def outputVector(index: int, totalLangs: int) -> np.array:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return np.array([float(v) for v in stringVec])

def main():
    with open("./data/meta/languages.txt","r") as file:
        languages = file.read()
        languages = languages.split(",")

    vecs = []
    for i,l in enumerate(languages):
        with open(f"./data/raw/{l}_text.txt","r") as file:
            vocab = Vocabulary(file.read())

        vocab.pruneVocabulary(10)
        vocabVec = vocab.vectorizeVocabulary(10)
        outVec = outputVector(i,10).reshape(1,10)

        combVec = np.concatenate((vocabVec,np.repeat(outVec,vocabVec.shape[0],axis=0)),axis=1)
        vecs.append(combVec)

    vecs = np.vstack(vecs)
    np.save("./data/processed/vectors.npy",vecs)

if __name__ == "__main__":
    main()