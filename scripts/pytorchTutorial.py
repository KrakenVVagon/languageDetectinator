import unicodedata
import string
from typing import Iterable, Tuple
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

def findFiles(path): return glob.glob(path)

allLetters = string.ascii_letters + " .,;'"
nChars = len(allLetters)

def unicodeToASCII(s: str, characters: Iterable[str]) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c)  != "Mn"
        and c in characters
    )

categories = []
categoryText = {}

def readNames(fileName: str) -> list:
    lines = open(fileName, encoding="utf-8").read().split(" ")
    return lines

for fileName in findFiles("data/processed/*.txt"):
    category = os.path.splitext(os.path.basename(fileName))[0]
    categories.append(category)
    lines = readNames(fileName)
    categoryText[category] = lines

nCategories = len(categories)

def letterToIndex(letter: str) -> str:
    return allLetters.find(letter)

def lineToTensor(line: str, nLetters: int) -> torch.TensorType:
    tensor = torch.zeros(len(line), 1, nLetters)
    for ind, letter in enumerate(line):
        tensor[ind][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def categoryFromOutput(output: torch.TensorType, categories: Iterable[str]) -> Tuple[str, int]:
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(categories: Iterable[str], categoryText: dict) -> Tuple[str, str, torch.TensorType, torch.TensorType]:
    category = randomChoice(categories)
    line = randomChoice(categoryText[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line, nChars)
    return category, line, category_tensor, line_tensor

n_hidden = 128
rnn = RNN(nChars, n_hidden, nCategories)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

def train(category_tensor, line_tensor, optimizer):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

n_iters = 100000
print_every = 5000

losses = []
currentLoss = 0

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(categories, categoryText)
    output, loss = train(category_tensor, line_tensor, optimizer)
    currentLoss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output, categories)
        correct = '✓' if guess == category else f'✗ {category}'
        print(f"{iter}, {iter/n_iters*100}%, {loss}, {line}, {guess}, {correct}")

    if iter % 1000 == 0:
        losses.append(currentLoss/1000)
        currentLoss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

confusion = torch.zeros(nCategories, nCategories)
n_confusion = 10000
correct = 0

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(categories, categoryText)
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output, categories)
    if guess == category:
        correct += 1

print(f"Overall accuracy: {correct/n_confusion*100:.4f}%")