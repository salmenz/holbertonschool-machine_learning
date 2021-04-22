#!/usr/bin/env python3
"""creates a convolutional autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    # encoder
    enc_in = K.Input(shape=input_dims)
    for i in range(len(filters)):
        if i == 0:
            y = (K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                 padding="SAME", activation="relu"))(enc_in)
            y = K.layers.MaxPool2D(pool_size=(2, 2), padding="SAME")(y)
        else:
            y = (K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                 padding="SAME", activation="relu"))(y)
            y = K.layers.MaxPool2D(pool_size=(2, 2), padding="SAME")(y)
    encoder = K.Model(enc_in, y)

    # decoder
    dec_in = K.Input(shape=latent_dims)
    for i in range(len(filters)-1, 0, -1):
        if i == len(filters)-1:
            y = (K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                 padding="SAME", activation="relu"))(dec_in)
            y = K.layers.UpSampling2D(size=(2, 2))(y)
        else:
            y = (K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                 padding="SAME", activation="relu"))(y)
            y = K.layers.UpSampling2D(size=(2, 2))(y)
    y = (K.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
         padding="VALID", activation="relu"))(y)
    y = K.layers.UpSampling2D(size=(2, 2))(y)

    y = (K.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
         activation="sigmoid", padding="SAME"))(y)
    decoder = K.Model(dec_in, y)

    out_encoder = encoder(enc_in)
    out_decoder = decoder(out_encoder)

    auto = K.Model(enc_in, out_decoder)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
