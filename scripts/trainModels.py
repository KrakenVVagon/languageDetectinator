"""Train and measure performance of different models.

"""
from languageDetectinator.models import LanguageDetector_RNN, RNN
import glob
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Iterable, Tuple
import random

def findFiles(path): return glob.glob(path)

def readNames(fileName: str) -> list:
    lines = open(fileName, encoding="utf-8").read().split(" ")
    return lines

def letterToIndex(letter: str) -> int:
    return ord(letter)-97

def lineToTensor(line: str) -> torch.TensorType:
    tensor = torch.zeros(len(line), 1, 26)
    for ind, letter in enumerate(line):
        tensor[ind][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output: torch.TensorType, categories: Iterable[str]) -> Tuple[str, int]:
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(categories: Iterable[str], categoryText: dict) -> Tuple[str, str, torch.TensorType, torch.TensorType]:
    category = randomChoice(categories)
    line = randomChoice(categoryText[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(model, category_tensor, line_tensor, optimizer, criterion):
    hidden = model.initHidden()

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

def evaluate(model, line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

categories = []
categoryText = {}

for fileName in findFiles("data/processed/*.txt"):
    category = os.path.splitext(os.path.basename(fileName))[0]
    categories.append(category)
    lines = readNames(fileName)
    categoryText[category] = lines

nCategories = len(categories)
n_hidden = 128
rnn = RNN(26, n_hidden, nCategories)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

n_iters = 100000
print_every = 5000

losses = []
currentLoss = 0

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(categories, categoryText)
    output, loss = train(rnn, category_tensor, line_tensor, optimizer, criterion)
    currentLoss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output, categories)
        correct = '✓' if guess == category else f'✗ {category}'
        print(f"{iter}, {iter/n_iters*100}%, {loss}, {line}, {guess}, {correct}")

    if iter % 1000 == 0:
        losses.append(currentLoss/1000)
        currentLoss = 0

n_confusion = 10000
correct = 0

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(categories, categoryText)
    output = evaluate(rnn, line_tensor)
    guess, guess_i = categoryFromOutput(output, categories)
    if guess == category:
        correct += 1

print(f"Overall accuracy: {correct/n_confusion*100:.4f}%")

# def validate(validationData: tuple, batch_size: int, criterion: nn.Module, model: nn.Module, hidden) -> tuple:
#     """Validation loop in order to get model accuracy while training. Can also be used for testing.
    
#     """
#     validationLoader = DataLoader(_loadData(validationData[0],validationData[1]), batch_size=batch_size, drop_last=True)

#     model.eval()
#     with torch.no_grad():
#         valLoss = 0
#         totalSamples = 0
#         correctSamples = 0

#         for i, (inputTensor, labelTensor) in enumerate(validationLoader):
#             inputTensor = torch.permute(inputTensor, (1, 0, 2))
#             inputTensor = inputTensor.to(device)
#             labelTensor = labelTensor.to(device)

#             validationOutputs, hidden = model(inputTensor, hidden)
#             valLoss += criterion(validationOutputs[-1], labelTensor).item()
#             _, predicted = torch.max(validationOutputs[-1].data, 1)
#             totalSamples += labelTensor.size(0)
#             correctSamples += (predicted == labelTensor).sum().item()

#         valLoss /= len(validationLoader)
#         valAccuracy = correctSamples / totalSamples

#     return (valLoss, valAccuracy)