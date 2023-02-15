from AIs.NeuralNetworks.MLPNeuralNetwork import SupervisedNeuralNetwork
import numpy as np
from AIs import TrainingGames

iterations_per_epoch = 1000
epochs = 10

best_hidden_weights = []
best_output_weights = []
lowest_loss = 999999999999

stored_error = []

config = {
    'input_activation_function' : 'relu',
    'hidden_activation_functions' : ['sigmoid', 'relu', 'sigmoid', 'relu'],
    'output_activation_function' : 'relu'
}

_neural_network = SupervisedNeuralNetwork(1, [4,4,4,4], 1, *config)

def print_weights() -> None:
    i = 1
    for hidden_layer in _neural_network.hidden_weights:
        print('Hidden Layer Weights ' + str(i))
        i += 1
        for weights in hidden_layer:
            print(weights)
    print('Output Layer Weights')
    for weights in _neural_network.output_weights:
        print(weights)

def print_biases() -> None:
    i = 1
    print('Input Layer Biases')
    for neuron in _neural_network.input_layer:
        print(neuron.bias)

    for hidden_layer in _neural_network.hidden_layers:
        print('Hidden Layer Biases ' + str(i))
        i += 1
        for neuron in hidden_layer:
            print(neuron.bias)

    print('Output Layer Biases')
    for neuron in _neural_network.output_layer:
        print(neuron.bias)

# Training Games
not_gate = TrainingGames.not_gate()

for epoch in range(epochs):
    for iteration in range(iterations_per_epoch):
        print('iteration: ' + str(iteration + (iterations_per_epoch * epoch) + 1))

        input_, actual = not_gate.run_game()
        _neural_network.network_train(input_, actual)
        stored_error.append(_neural_network.network_error)

print_weights()
print_biases()

_neural_network.show_error_line(stored_error, np.array(range(iterations_per_epoch * epochs)) + 1)

def use_network() -> None:
    input_ = input('input\n')
    prediction = _neural_network.network_use([int(input_)])
    print('predictionL: ' + str(prediction))

    use_network()

use_network()