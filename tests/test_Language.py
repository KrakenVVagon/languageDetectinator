from languageDetectinator.datasets import Language
import pytest

def test_generateVocabulary():
    lang = Language("en")
    with pytest.raises(TypeError):
        lang.generateTopics()

def test_generateTopics():
    lang = Language("en")
    assert len(lang.generateTopics(5)) == 5
    lang = Language("fr")
    assert len(lang.generateTopics(7)) == 7

def test_setVocabulary():
    lang = Language("en")
    lang.setVocabulary("my name is frobert")
    assert lang.vocabulary.text == "my name is frobert"
