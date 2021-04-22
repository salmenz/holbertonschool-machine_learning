#!/usr/bin/env python3
"""creates a variational autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    # encoder
    enc_in = K.Input(shape=(input_dims,))
    for i in range(len(hidden_layers)):
        if i == 0:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(enc_in)
        else:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(y)

    z_mean = K.layers.Dense(latent_dims)(y)
    z_log_sigma = K.layers.Dense(latent_dims)(y)

    def sampling(args):
        """new points sampling"""
        z_mean, z_log_sigma = args
        epsilon = K.backend.random_normal(shape=(K.backend.shape(z_mean)[0],
                                          latent_dims), mean=0., stddev=0.1)
        return z_mean + K.backend.exp(z_log_sigma) * epsilon

    z = K.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = K.Model(enc_in, [z_mean, z_log_sigma, z])

    # decoder
    dec_in = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers)-1, -1, -1):
        if i == len(hidden_layers)-1:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(dec_in)
        else:
            y = (K.layers.Dense(hidden_layers[i], activation="relu"))(y)
    y = (K.layers.Dense(input_dims, activation="sigmoid"))(y)
    decoder = K.Model(dec_in, y)

    # instantiate VAE model
    outputs = decoder(encoder(enc_in))
    vae = K.Model(enc_in, outputs)

    def loss(true, pred):
        """calculate loss"""
        reconstruction_loss = K.losses.binary_crossentropy(enc_in, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - K.backend.square(z_mean) - \
            K.backend.exp(z_log_sigma)
        kl_loss = K.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.backend.mean(reconstruction_loss + kl_loss)
    vae.compile(optimizer='adam', loss=loss)

    return encoder, decoder, vae
