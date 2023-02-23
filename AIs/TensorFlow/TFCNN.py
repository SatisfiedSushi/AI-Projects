import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

class SupervisedNeuralNetwork:
    def __init__(self, test_data, train_data, validation_data, test_validation_data):
        self.training_data = {
            'features': train_data,
            'actual': validation_data
        }
        self.test_data = {
            'features': test_data,
            'actual': test_validation_data
        }

        # how many categories in the images to detect
        self.num_classes = 5

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
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train_model(self):
        try:
            self.model.fit(
                self.training_data.get('features'),
                validation_data=self.training_data.get('actual'),
                epochs=3,
                )
        except:
            print('no training data')