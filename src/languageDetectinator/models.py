"""Deep learning models to learn and guess what language a word is

"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim

class LanguageDetector_CNN(nn.Module):

    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int) -> None:
        super(LanguageDetector_CNN,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.conv1 = nn.Conv1d(self.inputSize,self.hiddenSize,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(self.hiddenSize,self.hiddenSize,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(self.hiddenSize*32,512)
        self.fc2 = nn.Linear(512,self.outputSize)
        return None
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, self.hiddenSize * 32)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
class LanguageDetector_FFNN(nn.Module):

    def __init__(self, inputSize: int, outputSize: int, hiddenSize: int):
        super(LanguageDetector_FFNN,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.fc1 = nn.Linear(inputSize,hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        return None

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class ModelTrainer():

    def __init__(self, model: nn.Module):
        return None
    
    def _loadData(self):
        return None
    
    def train(epochs, inputs, labels, optimzier, criterion):
        return None