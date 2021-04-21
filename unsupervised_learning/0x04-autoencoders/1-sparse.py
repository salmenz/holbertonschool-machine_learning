#!/usr/bin/env python3
"""sparse autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """sparse autoencoder"""
    # encoder
    enc_in = K.Input(shape=(input_dims,))
    for i in range(len(hidden_layers)):
        if i == 0:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(enc_in)
        else:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(y)
    k_r = K.regularizers.l1(lambtha)
    y = (K.layers.Dense(latent_dims, activation="relu",
         kernel_regularizer=k_r))(y)
    encoder = K.Model(enc_in, y)

    # decoder
    dec_in = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers)-1, -1, -1):
        if i == len(hidden_layers)-1:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(dec_in)
        else:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(y)
    y = (K.layers.Dense(input_dims, activation="sigmoid"))(y)
    decoder = K.Model(dec_in, y)

    out_encoder = encoder(enc_in)
    out_decoder = decoder(out_encoder)

    auto = K.Model(enc_in, out_decoder)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
