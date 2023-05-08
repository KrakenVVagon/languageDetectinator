"""Dataset manipulation and creation for the models

"""
import wikipedia
from unidecode import unidecode

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
        
        vocabulary = []
        for topic in self.topics:
            page = wikipedia.WikipediaPage(topic)
            vocabulary += unidecode(page.content)
        
        self.vocabulary = Vocabulary(vocabulary)
        return self.vocabulary
    
class Vocabulary():

    def __init__(self, words: list) -> None:
        self.words = words
        return None
    
    def pruneVocabulary(self, n: int) -> list:
        """Removes duplicate words and words above the desired length
        
        """
        return list()

    def vectorizeVocabulary(self) -> list:
        """Converts the vocabulary into a vectorized form from the Latin alphabet (26 chars)
        
        """
        return list()