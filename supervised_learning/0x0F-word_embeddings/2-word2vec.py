#!/usr/bin/env python3
"""creates and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a gensim word2vec model"""
    model = Word2Vec(sentences=sentences, min_count=min_count, iter=iterations,
                     size=size, window=window, sg=cbow, seed=seed,
                     negative=negative, workers=workers)
    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
