#!/usr/bin/env python3
"""class Dataset that loads and preps a dataset for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class Dataset"""
    def __init__(self, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        # Data train
        self.data_train, info = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True,
                                    with_info= True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)

        bs = info.splits['train'].num_examples
        self.data_train = self.data_train.filter(lambda x,y: tf.math.logical_and(
            tf.size(x) <= self.max_len ,
            tf.size(y) <= self.max_len))

        self.data_train = self.data_train.cache().shuffle(bs)
        self.data_train = self.data_train.padded_batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(AUTOTUNE)

        #Data valid
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(lambda i,j: tf.math.logical_and(
            tf.size(i) <= max_len ,
            tf.size(j) <= max_len))
        self.data_valid = self.data_valid.padded_batch(batch_size)

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

    def tf_encode(self, pt, en):
        """Wraps the encoder into tensorflow operation"""
        pt_lang, en_lang = tf.py_function(func=self.encode, inp=[pt, en],
                                          Tout=(tf.int64, tf.int64))
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
