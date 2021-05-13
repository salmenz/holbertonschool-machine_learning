#!/usr/bin/env python3
"""calculates the cumulative n-gram BLEU score for a sentence"""
import numpy as np


def new_list(lista, N):
    """preparate the n-gram"""
    return [lista[i:i+N] for i in range(len(lista)-N+1)]


def cumulative_bleu(references, sentence, n):
    """calculates the cumulative n-gram BLEU score for a sentence"""
    ngrams = []
    for i in range(1, n+1):
        N_sentence = (new_list(sentence, i))
        N_references = []
        for ref in references:
            N_references.append(new_list(ref, i))
        len_sen = len(N_sentence)
        sum_count_clip = 0
        for word in N_sentence:
            count_clip = 0
            for ref in N_references:
                if ref.count(word) > count_clip:
                    count_clip = ref.count(word)
            sum_count_clip += count_clip
        ngrams.append(sum_count_clip/len_sen)
    ngrams = np.array(ngrams)
    min = len(sentence)
    for ref in references:
        if abs(len(ref) - len_sen) < min:
            min = abs(len(ref) - len_sen)
            closest_len = len(ref)
    bp = np.exp(1 - closest_len / len(sentence))
    if len_sen > closest_len:
        bp = 1
    return bp * np.exp(np.sum((1 / n) * np.log(ngrams)))
