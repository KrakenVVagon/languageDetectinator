from languageDetectinator.datasets import Vocabulary

def test_pruneVocabulary():
    vocab = Vocabulary("test TEST Test 123 7%33||24 wilko fulmination")
    assert sorted(vocab.pruneVocabulary(5)) == sorted(["test","wilko"])

def test_vectorizeVocabulary():
    vocab = Vocabulary("hello")
    vocab.pruneVocabulary(8)
    assert vocab.vectorizeVocabulary(8) == [
        "0000000100000000000000000000001000000000000000000000000000000001000000000000000000000000010000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    ]

def test_len_vectorizeVocabulary():
    vocab = Vocabulary("hello")
    vocab.pruneVocabulary(8)
    assert len(vocab.vectorizeVocabulary(12)[0]) == 26*12