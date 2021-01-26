#!/usr/bin/env python3
"""add train the model with learning rate decay"""

import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """add train the model with learning rate decay"""
    if validation_data:
        calbacks = []
        if learning_rate_decay:
            def decayed_learning_rate(step):
                return alpha / (1 + decay_rate * step)
            callback = k.callbacks.LearningRateScheduler(decayed_learning_rate,
                                                         verbose=1)
            calbacks.append(callback)
        if early_stopping:
            callback = k.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=patience,
                                                 verbose=verbose)
            calbacks.append(callback)
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, shuffle=shuffle,
                           validation_data=validation_data, callbacks=calbacks)
    else:
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           pochs=epochs, verbose=verbose, shuffle=shuffle,
                           validation_data=validation_data)
