"""Quick script to generate some basic data for us to start training some models with!

"""

from languageDetectinator.datasets import Language
import pandas as pd

def main():
    with open("./data/meta/languages.txt","r") as file:
        languages = file.read()
        languages = languages.split(",")

    topicDict = {}
    for i,l in enumerate(languages):
        lang = Language(l)
        lang.generateTopics(50)
        lang.generateVocabulary()
        topicDict[l] = lang.topics

        # save the text one language at a time (raw)
        with open(f"./data/raw/{l}_text.txt","w") as langFile:
            langFile.write(lang.vocabulary.text)

    topicDF = pd.DataFrame(topicDict)
    topicDF.to_csv("./data/raw/topics.csv",index=False)

if __name__ == "__main__":
    main()