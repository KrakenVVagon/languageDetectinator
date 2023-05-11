from languageDetectinator.datasets import Vocabulary
import numpy as np

def test_pruneVocabulary():
    vocab = Vocabulary("test TEST Test 123 7%33||24 wilko fulmination")
    assert sorted(vocab.pruneVocabulary(5)) == sorted(["test","wilko"])

def test_vectorizeVocabulary():
    vocab = Vocabulary("hello")
    vocab.pruneVocabulary(8)
    vec = "0000000100000000000000000000001000000000000000000000000000000001000000000000000000000000010000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    vec = [float(v) for v in vec]
    assert vocab.vectorizeVocabulary(8).tolist() == [vec]

def test_len_vectorizeVocabulary():
    vocab = Vocabulary("hello i am a taco")
    vocab.pruneVocabulary(8)
    assert len(vocab.vectorizeVocabulary(12)[0]) == 26*12
    assert vocab.vectorizeVocabulary(12).shape == (5,26*12)