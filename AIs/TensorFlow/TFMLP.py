import tensorflow as tf


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
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='sigmoid'),
        ])

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=self.loss_fn,
            metrics=['accuracy']
        )

    def train_model(self):
        try:
            self.model.fit(
                self.training_data.get('features'),
                self.training_data.get('actual'),
                epochs=10,
                batch_size=2000,
                validation_split=0.2
            )
        except:
            print('no training data')


    def evaluate_model(self):
        try:
            self.model.evaluate(
                self.test_data.get('features'),
                self.test_data.get('actual'),
                verbose=2
            )
        except:
            print('no test data')
