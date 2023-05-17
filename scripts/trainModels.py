"""Train and measure performance of different models.

"""

from languageDetectinator.models import LanguageDetector_CNN, LanguageDetector_FFNN, ModelTrainer
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split

def loadData(filePath):
    """Load some vectorized data to be fed into the model
    
    """
    data = np.load(filePath)
    inputs = data[:,:-10]
    labels = data[:,-10:]
    return inputs, labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs, labels = loadData("./data/processed/vectors.npy")
x_train, x_pre, y_train, y_pre = train_test_split(inputs,labels,test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_pre, y_pre, test_size=0.5)

ffnn_model = LanguageDetector_FFNN(260, 10)
ffnn_trainer = ModelTrainer(ffnn_model,device)
optimizer = optim.Adam(ffnn_model.parameters(), lr=0.001)

history = ffnn_trainer.train(200, inputs, labels, optimizer, nn.CrossEntropyLoss(), validation_data=(x_val,y_val))