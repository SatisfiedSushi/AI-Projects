import numpy as np

class Neuron:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.bias = 0

    def change_inputs(self, inputs: [float]) -> None:
        self.inputs = inputs

    def change_weights(self, weights: [float]) -> None:
        self.weights = weights

    def change_bias(self, bias: float) -> None:
        self.bias = bias

    def get_output(self) -> float:
        return np.dot(self.inputs, self.weights) + self.bias

class NeuralNetwork:
    def __init__(self, input_layer: int, hidden_layers: [int], output_layer: int):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        for i in range(input_layer):
            self.input_layer.append(Neuron())

        for i in hidden_layers:
            hidden_layer = []
            for j in range(i):
                hidden_layer.append(Neuron)

            self.hidden_layers.append(hidden_layer)

        for i in range(output_layer):
            self.output_layer.append(Neuron())
