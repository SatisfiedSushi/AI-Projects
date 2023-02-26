import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

class SupervisedNeuralNetwork:
    def __init__(self, test_data, train_data, validation_data):
        self.training_data = {
            'features': train_data,
            'actual': validation_data
        }
        self.test_data = {
            'features': test_data
        }

        # how many categories in the images to detect
        self.num_classes = 4

        # create CNN
        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])

    def compile_model(self):
        try:
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        except:
            print('failed to compile')


    def train_model(self):
        try:
            self.model.fit(
                self.training_data.get('features'),
                validation_data=self.training_data.get('actual'),
                epochs=10,
                )
        except:
            print('no training data')

    def test_model(self):
        try:
            print(self.model.predict(
                self.test_data.get('features')
            ))
        except:
            print('no training data')
