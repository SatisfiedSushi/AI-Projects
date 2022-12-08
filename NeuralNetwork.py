import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random as rndm

class Neuron:
    def dot(self, v1, v2):
        output = 0
        if type(v1) is list:
            for v in v1:
                output += v * v2[v1.index(v)]
        else:
            output += v1 * v2

        return output
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = ""
        self.error = 0

    def change_activation_function(self, activation_function: str) -> None:
        self.activation_function = activation_function

    def change_inputs(self, inputs: [float]) -> None:
        self.inputs = inputs

    def change_weights(self, weights: [float]) -> None:
        self.weights = weights

    def change_bias(self, bias: float) -> None:
        self.bias = bias

    def change_error(self, error) -> None:
        self.error = error

    def get_output(self) -> float:
        transposed_weights = self.weights
        reference_inputs = self.inputs
        output = self.dot(reference_inputs, transposed_weights) + self.bias
        return output

class ActivationFunctions:
    def __init__(self):
        self.forward = self.Forward()
        self.backward = self.Backward()
    class Forward:
        def relu(x):
            return np.maximum(0, x)

        def softmax(x):
            return np.exp(x) / sum(np.exp(x))

    class Backward:
        def relu(x):
            output = 0
            if x <= 0:
                output = 0
            else:
                output = 1
            return output


# Multilayer Perceptrons (MLPs) neural network model
class NeuralNetwork:
    def add_edge_to_graph(self, graph, e1, e2, c, w):
        graph.add_edge(e1, e2, color=c, weight=w)

    def state_position(self, points, graph, axis):
        color_list = ['red', 'blue', 'green', 'yellow', 'purple']
        positions = {point: point for point in points}
        edges = self.G.edges()
        nodes = self.G.nodes()

        edge_colors = [self.G[u][v]['color'] for u, v in edges]
        nx.draw(graph, pos=positions, node_size=0, edge_color=edge_colors, node_color='black', ax=axis)

    def set_random_weights(self, weights):
        return_weights = []
        for i in weights:
            return_weights.append(rndm.random())
        return return_weights

    def set_random_biases(self, biases):
        return rndm.random()

    def __init__(self, input_layer: int, hidden_layers: [int], output_layer: int):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        self.weights = []
        self.biases = []

        self.hidden_weights = []
        self.output_weights = []

        self.predicted_output = []
        self.network_error = []

        self.learning_rate = 0.7 # 0.7

        # networkx graphs
        self.G = nx.Graph

        # set input layer
        for i in range(input_layer):
            self.input_layer.append(Neuron([], 1, 0))

        # set hidden layers
        iteration = 0
        for i in hidden_layers:
            hidden_layer = []
            hidden_layer_weights = []
            for j in range(i):
                weights = []
                if iteration == 0:
                    weights = np.zeros(len(self.input_layer))
                else:
                    weights = np.zeros(len(self.hidden_layers[-1]))
                weights = np.clip(self.set_random_weights(weights), 1e-7, 1 - 1e-7)
                hidden_layer_weights.append(weights)
                hidden_layer.append(Neuron([], weights, 0))
            self.hidden_layers.append(hidden_layer)
            self.hidden_weights.append(hidden_layer_weights)

            iteration += 1

        # set output layer
        for i in range(output_layer):
            other_weights = np.clip(self.set_random_weights(np.zeros(len(self.hidden_layers[-1]))), 1e-7, 1 - 1e-7)
            self.output_layer.append(Neuron([], other_weights, 0))
            self.output_weights.append(other_weights)

        self.weights = [self.hidden_weights, self.output_weights]

    def set_input_layer(self, inputs: [int]):
        inputs_ = np.clip(inputs, 1e-7, 1 - 1e-7)
        for i in range(len(inputs)):
            self.input_layer[i].change_inputs(inputs_[i])

    def forward_pass(self, neuron: Neuron, activation_function: str) -> [int]:
        output = []

        # activation functions
        get_output = neuron.get_output()

        match activation_function:
            case "relu":
                output = ActivationFunctions.Forward.relu(get_output)
            case "softmax":
                output = ActivationFunctions.Forward.softmax(get_output)
            case _:
                output = ActivationFunctions.Forward.relu(get_output)

        return output

    def cost_function(self, output, actual):
        loss = 0
        # actual_ = np.clip(actual, 1e-7, 1 - 1e-7)
        for i in range(len(output)):
            loss_ = (output[i] - actual[i])**2
            loss += np.clip(loss_, 1e-7, 1 - 1e-7)

        '''loss = 0
        actual_ = np.clip(actual, 1e-7, 1 - 1e-7)
        for i in output:
            loss += (i * actual_[output.index(i)])
        return -np.log(loss)'''

        return loss

    def reset_inputs(self):
        for input_neuron in self.input_layer:
            input_neuron.change_inputs([])

        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron.change_inputs([])

        for output_neuron in self.output_layer:
            output_neuron.change_inputs([])

    def set_weights(self, hidden_weights, output_weights):
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights

        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron.change_weights(hidden_weights[self.hidden_layers.index(hidden_layer)][hidden_layer.index(hidden_neuron)])

        for output_neuron in self.output_layer:
            output_neuron.change_weights(output_weights[self.output_layer.index(output_neuron)])

    def network_forward_pass(self) -> []:
        # input layer -> first hidden layer
        for input_neuron in self.input_layer:
            input_neuron_output = self.forward_pass(input_neuron, "relu")
            for first_hidden_neuron in self.hidden_layers[0]:
                new_inputs = first_hidden_neuron.inputs.copy()
                new_inputs.append(input_neuron_output)
                first_hidden_neuron.change_inputs(new_inputs)
                first_hidden_neuron.change_activation_function("relu")

        # hidden layer -> next hidden layer
        if len(self.hidden_layers) > 1:
            for hidden_layer in self.hidden_layers:
                if hidden_layer != self.hidden_layers[-1]:
                    for hidden_neuron in hidden_layer:
                        hidden_neuron_output = self.forward_pass(hidden_neuron, "relu")
                        for next_hidden_neuron in self.hidden_layers[self.hidden_layers.index(hidden_layer) + 1]:
                            new_inputs = next_hidden_neuron.inputs.copy()
                            new_inputs.append(hidden_neuron_output)
                            next_hidden_neuron.change_inputs(new_inputs)
                            next_hidden_neuron.change_activation_function("relu")

        # last hidden layer -> output layer
        for last_hidden_neuron in self.hidden_layers[-1]:
            last_hidden_neuron_output = self.forward_pass(last_hidden_neuron, "relu")
            for output_neuron in self.output_layer:
                new_inputs = output_neuron.inputs.copy()
                new_inputs.append(last_hidden_neuron_output)
                output_neuron.change_inputs(new_inputs)
                output_neuron.change_activation_function("relu")

        # output layer -> output
        output = []
        for output_neuron in self.output_layer:
            output.append(self.forward_pass(output_neuron, "relu"))

        return output

    def backpropagate_error(self, neuron: Neuron, expected_=None):
        expected = np.clip(expected_, 1e-7, 1 - 1e-7)
        # Get the output of the neuron
        output = neuron.get_output()

        # Calculate the derivative of the activation function with respect to the output
        derivative_activation_function_output = ActivationFunctions.Backward.relu(int(output))
        # print('derivative_activation_function_output: ' + str(derivative_activation_function_output))
        if expected is not None:

            # Calculate the derivative of the error with respect to the output
            derivative_error = derivative_activation_function_output * (output - expected)

            # Update the error of the neuron
            neuron.change_error(derivative_error)

            #    Loop through the inputs and update the weights
            for i in range(len(neuron.inputs)):
                # Get the current weight and input value
                weight = neuron.weights[i]
                input_value = neuron.inputs[i]

                # Calculate the weight update
                weight_update = -self.learning_rate * derivative_error * input_value

                # Update the weight
                neuron.weights[i] = weight + weight_update

            # Update the bias
            neuron.bias = neuron.bias - self.learning_rate * derivative_error

    def network_backpropagate(self, expected) -> []:
        for output_neuron in self.output_layer:
            self.backpropagate_error(output_neuron, expected[self.output_layer.index(output_neuron)])

        # Propagate the error to the hidden layers
        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron_error = 0
                for output_neuron in self.output_layer:
                    hidden_neuron_error += output_neuron.error * output_neuron.weights[
                        hidden_layer.index(hidden_neuron)]
                self.backpropagate_error(hidden_neuron, hidden_neuron_error)

    def network_train(self, inputs, actual):
        self.reset_inputs()
        self.set_input_layer(inputs)
        self.predicted_output = self.network_forward_pass()
        self.network_backpropagate(actual)
        self.network_error = self.cost_function(self.predicted_output, actual)

    def show_error_line(self, stored_error, all_iterations):
        plt.scatter(all_iterations, stored_error)
        plt.plot(all_iterations, stored_error)
        plt.show()

    def show_neural_network(self) -> None:
        #nx.draw(self.G)
        plt.show()