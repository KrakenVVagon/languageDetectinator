"""Dataset manipulation and creation for the models

"""
import wikipedia
from unidecode import unidecode
import re
import numpy as np
from typing import Iterable
from torch.utils.data import TensorDataset
import torch

class languageDataset(TensorDataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        return None
    
    def __len__(self):
        return len(self.inputs)
    
    # we want this to return a value (label) and then a torch tensor for the inputs
    # tensor should have shape (len(input), numChars)
    def __getitem__(self, index):
        array = np.array(self.inputs[index])
        array = array.reshape(len(self.inputs[index]), 26)
        return torch.from_numpy(array).type(torch.float32), torch.tensor([self.labels[index]], dtype=torch.long)

class Vocabulary():

    def __init__(self, text: str) -> None:
        self.text = text
        return None
    
    def pruneVocabulary(self, n: int, duplicate: bool=False, keepAccents: bool=False) -> list:
        """Removes duplicate words and words above the desired length
        
        """
        subText = self.text.lower()
        if keepAccents:
            subText = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", subText)
        else:
            subText = re.sub(r"[^a-zA-Z\s]", "", subText)
        words = subText.split()

        self.words = []
        for word in words:
            if len(word) > n:
                continue
            self.words.append(word)
        
        if duplicate:
            return self.words
        self.words = list(set(self.words))
        return self.words

    def vectorizeVocabulary(self, n: int) -> np.array:
        """Converts the vocabulary into a vectorized form from the Latin alphabet (26 chars)
        
        """
        self.vectors = []
        for word in self.words:
            vec = ""
            for i,l in enumerate(word):
                ind = ord(l)-97
                vec += (str(0)*ind + str(1) + str(0)*(25-ind))
            excess = n-len(word)
            vec += str(0)*26*excess
            vec = [float(v) for v in vec]
            self.vectors.append(vec)
        
        self.vectors = np.array(self.vectors)
        return self.vectors
    
    def longVectorize(self, words: Iterable[str]=None) -> list:
        """Converst the vocabulary into a vectors of [len(n), 26]
        
        """
        words = words or self.words
        self.longVectors = []

        for word in words:
            wordVec = []
            for i,l in enumerate(word):
                ind = ord(l)-97
                vec = (str(0)*ind + str(1) + str(0)*(25-ind))
                wordVec.append([float(v) for v in vec])
            self.longVectors.append(wordVec)

        return self.longVectors

class Language():

    def __init__(self, language: str, topics: list=None, vocabulary: str=None) -> None:
        self.language = language
        self.topics = topics
        self.vocabulary = vocabulary
        wikipedia.set_lang(self.language)
        return None
    
    def generateTopics(self, n: int) -> list:
        """Generates n random topics from wikipedia in the specified language
        
        """
        wikipedia.set_lang(self.language)
        self.topics = wikipedia.random(n)
        return wikipedia.random(n)
    
    def generateVocabulary(self, topics: list=None, decodeLang: bool=True) -> Vocabulary:
        """Generate a Vocabulary object using text from Wikipedia articles
        
        """
        topics = topics or self.topics

        if topics is None:
            raise TypeError("Topics cannot be None. Must be iterable")
        
        vocabulary = ""
        for topic in topics:
            page = self._randomPageSelector(topic)

            if decodeLang:
                vocabulary += f"{unidecode(page.content)} "
            else:
                vocabulary += f"{page.content} "
        
        self.vocabulary = Vocabulary(vocabulary)
        return self.vocabulary
    
    def _randomPageSelector(self,topic):
        selection = False
        while not selection:
            # try and get the page but if it breaks we take a different random page
            try:
                print(f"Getting page for: {topic}")
                page = wikipedia.WikipediaPage(title=topic)
                selection = True
            except:
                topic = wikipedia.random(1)
        return page
    
    def setVocabulary(self, text: str) -> None:
        """Specify the set of words to use as the basis for the vocabulary
        
        """
        self.vocabulary = Vocabulary(text)
        return None