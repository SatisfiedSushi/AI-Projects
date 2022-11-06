import numpy as np
import matplotlib as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random as rndm

class Neuron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

    def change_inputs(self, inputs: [float]) -> None:
        self.inputs = inputs

    def change_weights(self, weights: [float]) -> None:
        self.weights = weights

    def change_bias(self, bias: float) -> None:
        self.bias = bias

    def get_output(self) -> float:
        print("inputs: " + str(self.inputs))
        print("weights: " + str(self.weights))
        output = np.dot(self.inputs, self.weights) + self.bias
        return output

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

    def __init__(self, input_layer: int, hidden_layers: [int], output_layer: int):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        self.weights = []
        self.biases = []

        # networkx graphs
        self.G = nx.Graph

        # set input layer
        for i in range(input_layer):
            self.input_layer.append(Neuron([], 1, 0))

        # set hidden layers
        for i in hidden_layers:
            hidden_layer = []
            for j in range(i):
                weights = []
                if i == hidden_layers[0]:
                    weights = np.zeros(len(self.input_layer))
                else:
                    weights = np.zeros(len(self.hidden_layers[-1]))
                weights = np.clip(self.set_random_weights(weights), 1e-7, 1 - 1e-7)
                hidden_layer.append(Neuron([], weights, 0))
            self.hidden_layers.append(hidden_layer)

        # set output layer
        for i in range(output_layer):
            self.output_layer.append(Neuron([], np.clip(self.set_random_weights(np.zeros(len(self.hidden_layers[-1]))), 1e-7, 1 - 1e-7), 0))

    def set_input_layer(self, inputs: [int]):
        inputs_ = np.clip(inputs, 1e-7, 1 - 1e-7)
        for i in range(len(inputs)):
            self.input_layer[i].change_inputs(inputs_[i])

    def forward_pass(self, neuron: Neuron, activation_function: str) -> [int]:
        output = []

        # activation functions
        def relu(x):
            return np.maximum(0, x)

        def softmax(x):
            return np.exp(x)/sum(np.exp(x))

        match activation_function:
            case "relu":
                output = relu(neuron.get_output())
            case "softmax":
                output = softmax(neuron.get_output())
            case _:
                output = relu(neuron.get_output())

        return output

    def calculate_loss(self, output, actual):
        loss = 0
        actual_ = np.clip(actual, 1e-7, 1 - 1e-7)
        for i in output:
            loss += (i * actual_[output.index(i)])
        return -np.log(loss)

    def reset_inputs(self):
        for input_neuron in self.input_layer:
            input_neuron.change_inputs([])

        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron.change_inputs([])

        for output_neuron in self.output_layer:
            output_neuron.change_inputs([])

    def get_weights(self):
        input_weights = []
        hidden_weights = []
        output_weights = []

        for input_neuron in self.input_layer:
            input_weights.append(input_neuron.weights)

        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron.change_inputs([])

        for output_neuron in self.output_layer:
            output_neuron.change_inputs([])

    def set_weights(self):
        pass

    def network_forward_pass(self) -> []:
        # input layer -> first hidden layer
        print("input layer:")
        for input_neuron in self.input_layer:
            input_neuron_output = self.forward_pass(input_neuron, "relu")
            for first_hidden_neuron in self.hidden_layers[0]:
                new_inputs = first_hidden_neuron.inputs.copy()
                new_inputs.append(input_neuron_output)
                first_hidden_neuron.change_inputs(new_inputs)

        # hidden layer -> next hidden layer
        if len(self.hidden_layers) > 1:
            for hidden_layer in self.hidden_layers:
                print("hidden layer " + str(self.hidden_layers.index(hidden_layer) + 1) + ":")
                if hidden_layer != self.hidden_layers[-1]:
                    for hidden_neuron in hidden_layer:
                        hidden_neuron_output = self.forward_pass(hidden_neuron, "relu")
                        for next_hidden_neuron in self.hidden_layers[self.hidden_layers.index(hidden_layer) + 1]:
                            new_inputs = next_hidden_neuron.inputs.copy()
                            new_inputs.append(hidden_neuron_output)
                            next_hidden_neuron.change_inputs(new_inputs)

        # last hidden layer -> output layer
        print("output layer:")
        for last_hidden_neuron in self.hidden_layers[-1]:
            last_hidden_neuron_output = self.forward_pass(last_hidden_neuron, "relu")
            for output_neuron in self.output_layer:
                new_inputs = output_neuron.inputs.copy()
                new_inputs.append(last_hidden_neuron_output)
                output_neuron.change_inputs(new_inputs)

        # output layer -> output
        output = []
        for output_neuron in self.output_layer:
            output.append(self.forward_pass(output_neuron, "relu"))

        return output

    def show_neural_network(self) -> None:
        nx.draw(self.G)
