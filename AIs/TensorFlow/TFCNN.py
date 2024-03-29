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
        self.num_classes = 15

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
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(self.num_classes)
        ])

    def compile_model(self):
        try:
            self.model.compile(
                optimizer='adam',
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        except:
            print('failed to compile')

    def train_model(self):
        try:
            checkpoint_path = 'AIs/TensorFlow/Checkpoints'
            checkpoint_dir = os.path.dirname(checkpoint_path)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            self.model.fit(
                self.training_data.get('features'),
                validation_data=self.training_data.get('actual'),
                epochs=10,
                callbacks=[cp_callback]
                )
        except:
            print('no training data')

    def test_model(self):
        try:
            self.model.predict(
                self.test_data.get('features')
            )
        except:
            print('no training data')

    def single_test_model(self, passed_test_dataset):
        try:
            print(self.model.predict(
                passed_test_dataset.get('features')
            ))
        except:
            print('no training data')