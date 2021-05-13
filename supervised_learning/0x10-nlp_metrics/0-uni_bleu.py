#!/usr/bin/env python3
"""calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence"""
    len_sen = len(sentence)
    min = max(len_sen, max(len(i) for i in references))
    sum_count_clip = 0
    for word in sentence:
        count_clip = 0
        for ref in references:
            if ref.count(word) > count_clip:
                count_clip = ref.count(word)
            if abs(len(ref) - len_sen) < min:
                min = abs(len(ref) - len_sen)
                closest_len = len(ref)
        sum_count_clip += count_clip
    bp = 1
    if len_sen > closest_len:
        bp = np.exp(1 - closest_len / len_sen)
    return bp * sum_count_clip / len_sen
