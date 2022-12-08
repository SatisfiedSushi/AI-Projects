from NeuralNetwork import NeuralNetwork
import numpy as np
from random import *

iterations_per_epoch = 1000
epochs = 10

best_hidden_weights = []
best_output_weights = []
lowest_loss = 999999999999

stored_error = []

_neural_network = NeuralNetwork(2, [4,4,4,4], 2)

# Training Games
def not_gate():
    input = [randint(0, 1), randint(0, 1)]
    actual = [0 if input[0] == 1 else 1, 0 if input[0] == 1 else 1]

    return input, actual

def no_change():
    input = [randint(0, 1), randint(0, 1)]
    actual = input.copy()
    return input, actual

for epoch in range(epochs):
    for iteration in range(iterations_per_epoch):
        print('iteration: ' + str(iteration + (iterations_per_epoch * epoch) + 1))

        input, actual = not_gate()
        _neural_network.network_train(input, actual)
        stored_error.append(_neural_network.network_error)

for layer in _neural_network.hidden_weights:
    print('Hidden Layer Weights')
    for weights in layer:
        print(weights)
print('Output Layer Weights')
for weights in _neural_network.output_weights:
    print(weights)

_neural_network.show_error_line(stored_error, np.array(range(iterations_per_epoch * epochs)) + 1)