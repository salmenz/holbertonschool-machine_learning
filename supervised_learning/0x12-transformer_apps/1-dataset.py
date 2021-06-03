#!/usr/bin/env python3
"""class Dataset that loads and preps a dataset for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class Dataset"""
    def __init__(self):
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)
        return token_pt, token_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        tok_pt = [self.tokenizer_pt.vocab_size]+self.tokenizer_pt.encode(
            pt.numpy())+[(self.tokenizer_pt.vocab_size) + 1]
        tok_en = [self.tokenizer_en.vocab_size]+self.tokenizer_en.encode(
            en.numpy())+[(self.tokenizer_en.vocab_size) + 1]
        return tok_pt, tok_en