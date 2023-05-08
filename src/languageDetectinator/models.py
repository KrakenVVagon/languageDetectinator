"""Deep learning models to learn and guess what language a word is

"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim

class LanguageDetector(nn.Module):

    def __init__(self) -> None:
        super(LanguageDetector,self).__init__()

        return None