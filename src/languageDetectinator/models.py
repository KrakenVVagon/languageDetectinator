"""Deep learning models to learn and guess what language a word is

"""
import torch
from torch import nn

class RNN(nn.Module):
    """Tutorial version of the RNN
    
    """
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
    """Character level RNN like the tutorial uses - edited to take a variety of hidden layer sizes
    
    """
    def __init__(self, inputSize: int, outputSize: int, hiddenSizes: list):
        super(LanguageDetector_RNN,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSizes = hiddenSizes

        # first layer should be (input+last, n) for sizes
        # other hidden layers
        self.hiddenLayers = nn.ModuleList()
        for i, size in enumerate(self.hiddenSizes):
            if i==0:
                self.hiddenLayers.append(nn.Linear(self.inputSize + self.hiddenSizes[-1], size))
            else:
                self.hiddenLayers.append(nn.Linear(self.hiddenSizes[i-1], size))
        self.output = nn.Linear(self.inputSize + self.hiddenSizes[-1], self.outputSize)
        self.softmax = nn.LogSoftmax(dim=1)

        return None

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden[-1]), 1)

        hiddenStates = []
        for i, hiddenLayer in enumerate(self.hiddenLayers):
            if i==0:
                newHidden = hiddenLayer(combined)
            else:
                newHidden = hiddenLayer(hidden[i-1])
            hiddenStates.append(newHidden)

        output = self.output(combined)
        output = self.softmax(output)

        return output, hiddenStates
    
    def initHidden(self):
        self.hidden_states = []
        for size in self.hiddenSizes:
            self.hidden_states.append(torch.zeros(1, size))
        return self.hidden_states