import tensorflow as tf
from tensorflow.keras import layers, models


def create_genomic_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = create_genomic_model(1000)  # Replace 1000 with the number of features in your data
    model.summary()
