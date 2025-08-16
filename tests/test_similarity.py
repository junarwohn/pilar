import random
import time
from difflib import SequenceMatcher
from rapidfuzz.distance import Levenshtein


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def rf_ratio(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b)


def test_accuracy_and_performance():
    pairs = [
        ("hello", "hello"),
        ("hello", "hallo"),
        ("apple", "apple"),
        ("apple", "banana"),
        ("", ""),
        ("abcdefgh", "abcxefgh"),
    ]
    for a, b in pairs:
        assert abs(seq_ratio(a, b) - rf_ratio(a, b)) < 0.01

    random.seed(0)
    sample = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10)) for _ in range(1000)]
    pairs = list(zip(sample, reversed(sample)))

    start = time.perf_counter()
    for a, b in pairs:
        seq_ratio(a, b)
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    for a, b in pairs:
        rf_ratio(a, b)
    rf_time = time.perf_counter() - start

    # Rapidfuzz should be faster than SequenceMatcher
    assert rf_time < seq_time
