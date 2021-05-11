#!/usr/bin/env python3
"""creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding"""
    vec = TfidfVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)
    features = vec.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
