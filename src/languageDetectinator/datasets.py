"""Dataset manipulation and creation for the models

"""
import wikipedia
from unidecode import unidecode
import re
import numpy as np

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
        return list(set(self.words))

    def vectorizeVocabulary(self, n: int, flat: bool=True) -> np.array:
        """Converts the vocabulary into a vectorized form from the Latin alphabet (26 chars)
        
        """
        self.vectors = []
        for word in self.words:
            if not flat:
                self.vectors.append(self.longVectorize(n, word))
                continue
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
    
    def longVectorize(self, n: int, word: str) -> np.array:
        """Converst the vocabulary into a vectors of [len(n), 1, 26]
        
        """
        vectors = []
        for i,l in enumerate(word):
            ind = ord(l)-97
            vec = (str(0)*ind + str(1) + str(0)*(25-ind))
            vectors.append([float(v) for v in vec])

        vec = np.array(vectors)
        excess = n - len(vec)
        z = np.zeros((excess,26))
        vec = np.vstack([vec,z])

        return vec.reshape(len(vec),1,26)

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