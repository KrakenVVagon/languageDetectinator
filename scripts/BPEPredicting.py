from tokenizers import Tokenizer
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

def outputVector(index: int, totalLangs: int) -> list:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return [int(v) for v in stringVec]

with open("data/meta/languages.txt","r") as lanF:
    langs = lanF.read().split(",")

tokenizer = Tokenizer.from_file("models/tokenizer.json")
tokenizer.enable_padding(length=15)

tokens = []
for i, lang in enumerate(langs):

    with open(f"data/processed/{lang}_pruned.txt","r",encoding="utf-8") as f:
        words = f.read().split()

    outVec = outputVector(i, len(langs))
    tokens += [tokenizer.encode(word).ids+outVec for word in words]

tokens = np.array(tokens)
print(tokens.shape)
np.save("./data/processed/tokenVectors.npy", tokens)