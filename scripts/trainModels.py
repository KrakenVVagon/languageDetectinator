"""Train and measure performance of different models.

"""

from languageDetectinator.models import LanguageDetector_RNN, LanguageDetector_FFNN, ModelTrainer
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def loadData(filePath, langNum):
    """Load some vectorized data to be fed into the model
    
    """
    data = np.load(filePath)
    inputs = data[:,:-langNum]
    labels = data[:,-langNum:]
    return inputs, labels

def _loadData(inputs: np.array, labels: np.array) -> TensorDataset:
    """Transform the data from numpy arrays into PyTorch tensors.
    
    """
    inputTensor = torch.from_numpy(inputs).float()
    labelTensor = torch.from_numpy(labels).long()

    return TensorDataset(inputTensor, labelTensor)

def validate(validationData: tuple, batch_size: int, criterion: nn.Module, model: nn.Module, hidden) -> tuple:
    """Validation loop in order to get model accuracy while training. Can also be used for testing.
    
    """
    validationLoader = DataLoader(_loadData(validationData[0],validationData[1]), batch_size=batch_size, drop_last=True)

    model.eval()
    with torch.no_grad():
        valLoss = 0
        totalSamples = 0
        correctSamples = 0

        for i, (inputTensor, labelTensor) in enumerate(validationLoader):
            inputTensor = torch.permute(inputTensor, (1, 0, 2))
            inputTensor = inputTensor.to(device)
            labelTensor = labelTensor.to(device)

            # if inputTensor.size()[1] < hidden.size()[1]:
            #     hidden = hidden[:, :inputTensor.size()[1], :]
            
            # if inputTensor.size()[1] > hidden.size()[1]:
            #     hidden = hidden.expand(-1, inputTensor.size()[1], -1)

            validationOutputs, hidden = model(inputTensor, hidden)
            valLoss += criterion(validationOutputs[-1], labelTensor).item()
            _, predicted = torch.max(validationOutputs[-1].data, 1)
            totalSamples += labelTensor.size(0)
            correctSamples += (predicted == labelTensor).sum().item()

        valLoss /= len(validationLoader)
        valAccuracy = correctSamples / totalSamples

    return (valLoss, valAccuracy)

def train(model: nn.Module, epochs: int, inputs: np.array, labels: np.array, optimzier: optim.Optimizer, criterion: nn.Module, batch_size: int=32, validation_data: tuple=None) -> list:
    trainLoader = DataLoader(_loadData(inputs,labels),batch_size=batch_size, drop_last=True)

    history = []
    for e in range(epochs):
        epochLoss = 0
        for i, (inputTensor, labelTensor) in enumerate(trainLoader):
            inputTensor = torch.permute(inputTensor, (1, 0, 2))
            inputTensor = inputTensor.to(device)
            labelTensor = labelTensor.to(device)

            hidden = torch.zeros(12, inputTensor.size()[1], 50)

            optimzier.zero_grad()
            outputTensor, hidden =  model(inputTensor, hidden)
            loss = criterion(outputTensor[-1],labelTensor)
            loss.backward()
            optimzier.step()

            epochLoss += loss.item()

        valLoss, valAccuracy = None, None
        if validation_data is not None:
            valLoss, valAccuracy = validate(validation_data, batch_size, criterion, model, hidden)
            
        epochLoss /= len(trainLoader)
        history.append((epochLoss, valLoss, valAccuracy))
        print(f"Epoch: {e+1}; loss: {epochLoss:.4f}; valLoss: {valLoss:.4f}; valAcc: {valAccuracy:.4f}")

    return history

langNum = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inputs, labels = loadData("./data/processed/tokenVectors.npy", langNum)
# x_train, x_pre, y_train, y_pre = train_test_split(inputs,labels,test_size=0.2)
# x_val, x_test, y_val, y_test = train_test_split(x_pre, y_pre, test_size=0.5)

# ffnn_model = LanguageDetector_FFNN(15, langNum)
# ffnn_trainer = ModelTrainer(ffnn_model,device)
# optimizer = optim.Adam(ffnn_model.parameters(), lr=0.005)

#history = ffnn_trainer.train(100, inputs, labels, optimizer, nn.CrossEntropyLoss(), validation_data=(x_val,y_val), batch_size=1024)

rnn = LanguageDetector_RNN(26, 5)
optimizer = optim.SGD(rnn.parameters(), lr=0.005)
inputs = np.load("./data/processed/tokenVectors.npy")
labels = np.load("./data/processed/tokenLabels.npy")

x_train, x_pre, y_train, y_pre = train_test_split(inputs,labels,test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_pre, y_pre, test_size=0.5)

history = train(rnn, 100, x_train, y_train, optimizer, nn.NLLLoss(), batch_size=64, validation_data=(x_val,y_val))