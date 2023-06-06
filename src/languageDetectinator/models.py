"""Deep learning models to learn and guess what language a word is

"""
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

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
    
class LanguageDetector_RNN(nn.Module):
    """Character level RNN like the tutorial uses
    
    """
    def __init__(self, inputSize: int, outputSize: int, hiddenSizes: list):
        super(LanguageDetector_RNN,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSizes = hiddenSizes

        self.hidden1 = nn.Linear(self.inputSize + 1024, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 512)
        self.hidden4 = nn.Linear(512, 1024)
        self.output = nn.Linear(self.inputSize + 1024, self.outputSize)
        self.softmax = nn.LogSoftmax(dim=2)

        return None

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden[-1]), 2)

        h1 = self.hidden1(combined)
        h2 = self.hidden2(h1)
        h3 = self.hidden3(h2)
        h4 = self.hidden4(h3)
        hidden = [h1, h2, h3, h4]

        output = self.output(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self, sequence_length: int=1, batch_size: int=1):
        hidden_states = []
        for size in self.hiddenSizes:
            hidden_states.append(torch.zeros(sequence_length, batch_size, size))
        return hidden_states