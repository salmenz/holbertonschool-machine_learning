#!/usr/bin/env python3
"""analyze validaiton data"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """analyze validaiton data"""

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, validation_data=validation_data,
                       shuffle=shuffle)
