
#!/usr/bin/env python3

""" contains uni_bleu function"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU
    score for a sentence
    """
    c = len(sentence)
    r = np.array([len(r) for r in references])
    r = np.argmin(np.abs(r - c))
    r = len(references[r])
    bp = 1
    if r > c:
        bp = np.exp(1 - r / c)
    words = {}
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = sum(words.values()) / c
    return bp * p
