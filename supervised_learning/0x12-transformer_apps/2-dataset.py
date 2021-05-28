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
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)
        return token_pt, token_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        p = self.tokenizer_pt.vocab_size
        e = self.tokenizer_en.vocab_size
        pt_tokens = [p] + self.tokenizer_pt.encode(pt.numpy()) + [p + 1]
        en_tokens = [e] + self.tokenizer_pt.encode(en.numpy()) + [e + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wraps the encoder into tensorflow operation"""
        pt_lang, en_lang = tf.py_function(func=self.encode, inp=[pt, en],
                                          Tout=(tf.int64, tf.int64))
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
