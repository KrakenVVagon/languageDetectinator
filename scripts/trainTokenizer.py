from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"])
tokenizer.pre_tokenizer = Whitespace()

with open("data/meta/languages.txt","r") as lanF:
    langs = lanF.read().split(",")

corpi = []
for lang in langs:
    corpi.append(f"data/processed/{lang}_pruned.txt")

tokenizer.train(corpi,trainer=trainer)
tokenizer.save("models/tokenizer.json")