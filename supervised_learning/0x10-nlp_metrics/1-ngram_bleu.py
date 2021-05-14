#!/usr/bin/env python3
"""calculates the n-gram BLEU score for a sentence"""
import numpy as np


def new_list(lista, N):
    """preparate the n-gram"""
    return [lista[i:i+N] for i in range(len(lista)-N+1)]


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    N_sentence = (new_list(sentence, n))
    N_references = []
    for ref in references:
        N_references.append(new_list(ref, n))
    len_sen = len(N_sentence)
    min = max(len(sentence), max(len(i) for i in references))
    for ref in references:
        if abs(len(ref) - len(sentence)) < min:
            min = abs(len(ref) - len(sentence))
            closest_len = len(ref)
    sum_count_clip = 0
    added = []
    for word in N_sentence:
        count_clip = 0
        for ref in N_references:
            if ref.count(word) > count_clip:
                count_clip = ref.count(word)
        if word not in added:
            sum_count_clip += count_clip
            added.append(word)
    bp = np.exp(1 - closest_len / len(sentence))
    if len_sen > closest_len:
        bp = 1
    return bp * sum_count_clip / len_sen
