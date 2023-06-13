"""Train and measure performance of different models.

"""
from languageDetectinator.models import LanguageDetector_RNN
from languageDetectinator.datasets import Vocabulary, languageDataset
import glob
import os
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def findFiles(path): return glob.glob(path)

def trainStep(model, category_tensor, line_tensor, optimizer, criterion):
    hidden = model.initHidden()

    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor[0])
    loss.backward()
    optimizer.step()

    return output, loss.item()

def validate(model, validationData, criterion):
    validationDataset = languageDataset(validationData[0], validationData[1])
    validationLoader = DataLoader(validationDataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        correctSamples = 0
        valLoss = 0
        for _, (inputs, labels) in enumerate(validationLoader):
            inputs = torch.permute(inputs, (1, 0, 2))
            output = evaluate(model, inputs)
            _, predictionIndex = torch.max(output.data, 1)
            valLoss += criterion(output, labels[0]).item()

            if predictionIndex == labels[0]:
                correctSamples += 1

        valLoss /= len(validationLoader)
        valAccuracy = correctSamples / len(validationLoader)

    return valLoss, valAccuracy

def train(model, epochs, category_tensor, line_tensor, optimizer, criterion, validationData=None):
    history = []
    dataset = languageDataset(line_tensor, category_tensor)
    trainLoader = DataLoader(dataset, batch_size=1)

    for e in range(epochs):
        epochLoss = 0
        for _, (inputs, labels) in enumerate(trainLoader):
            inputs = torch.permute(inputs, (1, 0, 2))
            _, stepLoss = trainStep(model, labels, inputs, optimizer, criterion)
            epochLoss += stepLoss
        
        epochLoss /= len(trainLoader)
        history.append(epochLoss)

        valLoss, valAccuracy = 0, 0
        if validationData is not None:
            valLoss, valAccuracy = validate(model, validationData, criterion)

        print(f"Epoch: {e+1}; loss: {epochLoss:.4f}; valLoss: {valLoss:.4f}; valAcc: {valAccuracy:.4f}")

    return history

def evaluate(model, line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

languages = []
languageIds = []
languageVectors = []

for languageFile in findFiles("data/raw/*.txt"):
    languageName = os.path.splitext(os.path.basename(languageFile))[0]
    languages.append(languageName)
    languageText = open(languageFile, "r", encoding="utf-8").read()
    languageVocab = Vocabulary(languageText)
    words = languageVocab.pruneVocabulary(12, duplicate=False)
    languageVectors += languageVocab.longVectorize(words=words)
    languageIds += [len(languages)-1]*len(words)

x_train, testX, y_train, testY = train_test_split(languageVectors, languageIds, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(testX, testY, test_size=0.5)

n_hidden = 128
rnn = LanguageDetector_RNN(26, len(languages), [128])
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

history = train(rnn, 10, y_train, x_train, optimizer, criterion, validationData=(x_val, y_val))

testDataset = languageDataset(x_test, y_test)
correct = 0
testLoader = DataLoader(testDataset, batch_size=1)

for _, (line_tensor, category_tensor) in enumerate(testLoader):
    line_tensor = torch.permute(line_tensor, (1, 0, 2))
    output = evaluate(rnn, line_tensor)
    _, predictionIndex = torch.max(output.data, 1)
    if predictionIndex == category_tensor[0]:
        correct += 1

print(f"Overall testing accuracy: {correct/len(testLoader)*100:.4f}%")