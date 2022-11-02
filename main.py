import AIs.GateGuesser.GateGuesser as GateGuesser
import NeuralNetwork

# GateGuesser.GateGuesser()
NN = NeuralNetwork.NeuralNetwork(2, [2, 2], 2)
NN.set_input_layer([-0.7, 0.4])
NN.network_forward_pass()

for i in NN.input_layer:
    print(i.inputs)

for i in NN.hidden_layers[0]:
    print(i.inputs)

for i in NN.output_layer:
    print(i.inputs)
















