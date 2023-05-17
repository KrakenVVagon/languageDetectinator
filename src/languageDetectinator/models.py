"""Deep learning models to learn and guess what language a word is

"""
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

class LanguageDetector_CNN(nn.Module):
    """CNN version of a language detector network
    
    """
    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int) -> None:
        super(LanguageDetector_CNN,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.conv1 = nn.Conv1d(self.inputSize,self.hiddenSize,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(self.hiddenSize,self.hiddenSize,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=1)
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
    """More traditional FFNN language detector network
    
    """
    def __init__(self, inputSize: int, outputSize: int):
        super(LanguageDetector_FFNN,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.fc1 = nn.Linear(self.inputSize, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, self.outputSize)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        return None

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class ModelTrainer():
    """General framework to train PyTorch models
    
    """
    def __init__(self, model: nn.Module, device: torch.DeviceObjType):
        self.model = model
        self.device = device
        return None
    
    def _loadData(self, inputs: np.array, labels: np.array) -> TensorDataset:
        """Transform the data from numpy arrays into PyTorch tensors.
        
        """
        inputTensor = torch.from_numpy(inputs).float()
        labelTensor = torch.from_numpy(labels)
        return TensorDataset(inputTensor, labelTensor)
    
    def validate(self, validationData: tuple, batch_size: int, criterion: nn.Module) -> tuple:
        """Validation loop in order to get model accuracy while training. Can also be used for testing.
        
        """
        validationLoader = DataLoader(self._loadData(validationData[0],validationData[1]), batch_size=batch_size)

        self.model.eval()
        with torch.no_grad():
            valLoss = 0
            totalSamples = 0
            correctSamples = 0

            for i, (inputTensor, labelTensor) in enumerate(validationLoader):
                inputTensor = inputTensor.to(self.device)
                labelTensor = labelTensor.to(self.device)

                validationOutputs = self.model(inputTensor)
                valLoss += criterion(validationOutputs, labelTensor).item()
                _, predicted = torch.max(validationOutputs.data, 1)
                _, accLabels = torch.max(labelTensor.data, 1)
                totalSamples += labelTensor.size(0)
                correctSamples += (predicted == accLabels).sum().item()

            valLoss /= len(validationLoader)
            valAccuracy = correctSamples / totalSamples

        return (valLoss, valAccuracy)
    
    def train(self, epochs: int, inputs: np.array, labels: np.array, optimzier: optim.Optimizer, criterion: nn.Module, batch_size: int=32, validation_data: tuple=None) -> list:
        trainLoader = DataLoader(self._loadData(inputs,labels),batch_size=batch_size)

        history = []
        for e in range(epochs):
            epochLoss = 0
            for i, (inputTensor, labelTensor) in enumerate(trainLoader):
                inputTensor = inputTensor.to(self.device)
                labelTensor = labelTensor.to(self.device)

                optimzier.zero_grad()
                outputTensor =  self.model(inputTensor)
                loss = criterion(outputTensor,labelTensor)
                loss.backward()
                optimzier.step()

                epochLoss += loss.item()

            if validation_data is not None:
                valLoss, valAccuracy = self.validate(validation_data, batch_size, criterion)
                
            epochLoss /= len(trainLoader)
            history.append((epochLoss, valLoss, valAccuracy))
            print(f"Epoch: {e+1}; loss: {epochLoss:.4f}; valLoss: {valLoss:.4f}; valAcc: {valAccuracy:.4f}")

        return history