#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset:
"""
import tensorflow.keras as K
import tensorflow as tf


# preprocess function for the inputs
def preprocess_data(X, Y):
    """
    a function that trains a convolutional neural network to classify the
    CIFAR 10 dataset
    :param X: X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    """ main program """
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    print((X_train.shape, Y_train.shape))
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)
    print((X_train.shape, Y_train.shape))

    """include_top=False allows feature extraction by removing the last dense
    layers. This let us control the output and input of the model."""
    input_t = K.Input(shape=(224, 224, 3))
    res_model = K.applications.ResNet50(include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_t)
    # freeze some lyers
    for layer in res_model.layers[:-32]:
        layer.trainable = False

    # Check the freezed was done ok
    for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "-", layer.trainable)

    # improve the model
    model = K.models.Sequential()
    model.add(K.layers.Lambda(lambda image: tf.image.resize(image,
                                                            (224, 224))))
    model.add(res_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    # This callback saves the best weights obtained in the training
    check_point = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                              monitor="val_accuracy",
                                              mode="max",
                                              save_best_only=True,
                                              )

    # compile model and using RMSprop as optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[check_point])

    # the summary of the model
    model.summary()

    # saving model
    model.save("cifar10.h5")
