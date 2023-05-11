"""Dataset manipulation and creation for the models

"""
import wikipedia
from unidecode import unidecode
import re

class Language():

    def __init__(self, language: str, topics: list=None, vocabulary: list=None) -> None:
        self.language = language
        self.topics = topics
        self.vocabulary = vocabulary
        return None
    
    def generateTopics(self, n: int) -> list:
        self.topics = wikipedia.random(n)
        return wikipedia.random(n)
    
    def generateVocabulary(self) -> list:
        """Generate a Vocabulary object
        
        """
        if self.topics is None:
            raise TypeError("Topics cannot be None. Must be iterable")
        
        vocabulary = ""
        for topic in self.topics:
            page = wikipedia.WikipediaPage(topic)
            vocabulary += f"{unidecode(page.content)} "
        
        self.vocabulary = Vocabulary(vocabulary)
        return self.vocabulary
    
class Vocabulary():

    def __init__(self, text: str) -> None:
        self.text = text
        return None
    
    def pruneVocabulary(self, n: int) -> list:
        """Removes duplicate words and words above the desired length
        
        """
        subText = self.text.lower()
        subText = re.sub(r"[^a-zA-Z\s]", "", subText)
        words = subText.split()

        self.words = []
        for word in words:
            if len(word) > n:
                continue
            self.words.append(word)
        
        self.words = list(set(self.words))
        return self.words

    def vectorizeVocabulary(self, n: int) -> list:
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
            self.vectors.append(vec)
        return self.vectors