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

cnn_model = LanguageDetector_FFNN(260, 10)
cnn_trainer = ModelTrainer(cnn_model,device)

history = cnn_trainer.train(200, inputs, labels, optim.Adam(cnn_model.parameters(), lr=0.001), nn.CrossEntropyLoss())