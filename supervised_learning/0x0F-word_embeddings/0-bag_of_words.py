#!/usr/bin/env python3
"""creates a bag of words embedding matrix"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    vec = CountVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)
    features = vec.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
