import numpy as np

class GateGuesser:
    def __init__(self):
        self.a = np.array([0, 0, 1, 1])
        self.b = np.array([0, 1, 0, 1])

        self.total_input = [self.a, self.b]
        self.total_input = np.array(self.total_input)

        self.epochs = 5000
        self.learning_rate = 0.1
        self.input_neurons, hidden_neurons_1, output_neurons = 2, 2, 1

        self.input_layers = []

        self.biases = np.zeros(3) # b
        self.weights = [] # w
        self.pre_activations = [] # a
        self.activations = [] # h
        # self.w1 = np.random.rand(self.hidden_neurons_1, self.input_neurons)
        # self.w2 = np.random.rand(self.output_neurons, self.hidden_neurons_1)
        # print(self.w1)
        # print(self.w2)
        # 2-layer model

    def sigmoid(self, x) -> int:
        return 1/(1+np.exp(-x))

    def calculate_activation(self, weights, bias):
        pass

    def calculate_preactivation(self, weights, bias):


    def forward_propagation(self, x):
        pass