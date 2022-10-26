import AIs.GateGuesser.GateGuesser as GateGuesser
import NeuralNetwork

# GateGuesser.GateGuesser()
NN = NeuralNetwork.NeuralNetwork(2, [2, 2], 2)
print(NN.input_layer)
print(NN.hidden_layers)
print(NN.output_layer)

print(NN.hidden_layers[0][0].weights)



















