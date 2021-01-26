#!/usr/bin/env python3
"""train the model using early stopping"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """train the model using early stopping"""
    if validation_data and early_stopping:
        callback = k.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             verbose=verbose)
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose,
                           validation_data=validation_data,
                           shuffle=shuffle, callbacks=[callback])
    else:
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose,
                           validation_data=validation_data, shuffle=shuffle)
