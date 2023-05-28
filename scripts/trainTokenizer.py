from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np

def outputVector(index: int, totalLangs: int) -> np.array:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return np.array([float(v) for v in stringVec])

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[PAD]"])
tokenizer.pre_tokenizer = Whitespace()

with open("data/meta/languages.txt","r") as lanF:
    langs = lanF.read().split(",")

corpi = []
for lang in langs:
    corpi.append(f"data/processed/{lang}_pruned.txt")

tokenizer.train(corpi,trainer=trainer)
tokenizer.save("models/tokenizer.json")