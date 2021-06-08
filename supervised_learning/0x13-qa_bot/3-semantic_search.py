#!/usr/bin/env python3
"""perform semantic search on a corpus of documents"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """perform semantic search on a corpus of documents"""
    reference = [sentence]
    files = os.listdir(corpus_path)
    for file in files:
        if not file.endswith('.md'):
            continue
        with open(corpus_path + '/' + file, 'r', encoding='utf-8') as f:
            reference.append(f.read())
    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')

    embeddings = embed(reference)
    corr = np.inner(embeddings, embeddings)
    close = np.argmax(corr[0, 1:])

    return reference[close + 1]
