import numpy as np
import matplotlib as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

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
        output = np.dot(self.inputs, self.weights) + self.bias
        self.value = output
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
            return_weights.append(np.random.randint(0,5))
        return return_weights

    def __init__(self, input_layer: int, hidden_layers: [int], output_layer: int):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        # networkx graphs
        self.G = nx.Graph

        # set input layer
        for i in range(input_layer):
            self.input_layer.append(Neuron([], np.ones(len(self.hidden_layers[-1])), 0))

        # set hidden layers
        for i in hidden_layers:
            hidden_layer = []
            for j in range(i):
                weights = []
                if i == hidden_layers[0]:
                    weights = np.zeros(len(self.input_layer))
                else:
                    weights = np.zeros(len(self.hidden_layers[-1]))
                weights = self.set_random_weights(weights)
                hidden_layer.append(Neuron([], weights, 0))
            self.hidden_layers.append(hidden_layer)

        # set output layer
        for i in range(output_layer):
            self.output_layer.append(Neuron([], self.set_random_weights(np.zeros(len(self.hidden_layers[-1]))), 0))

    def set_input_layer(self, inputs: [int]):
        for i in range(len(inputs)):
            self.input_layer[i].change_inputs(inputs[i])
    def forward_pass(self, neuron: Neuron):
        output = np.maximum(0, neuron.get_output())
        return output

    def calculate_loss(self):
        pass

    def show_neural_network(self) -> None:
        nx.draw(self.G)
