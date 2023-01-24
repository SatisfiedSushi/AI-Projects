import numpy as np

class Forward:
    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

class Backward:
    def relu(x):
        output = 0
        if x <= 0:
            output = 0
        else:
            output = 1
        return output

    def sigmoid(x):
        return np.multiply(1 / (1 + np.exp(-x)), 1 - (1 / (1 + np.exp(-x))))