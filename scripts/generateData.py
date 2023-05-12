"""Quick script to generate some basic data for us to start training some models with!

"""

from languageDetectinator.datasets import Language

def main():
    languages = ["en","fr","de","es","cs","pt","pl","ar","ru","ja"]

    for i,l in enumerate(languages):
        lang = Language(l)
        lang.generateTopics(5)
        lang.generateVocabulary()

if __name__ == "__main__":
    for k in range(20):
        main()