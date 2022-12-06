import NeuralNetwork
import numpy as np
import random

'''x = random.randint(0,1)
y = random.randint(0,1)
print("x: " + str(x))
print("y: " + str(y))

actual = []
if x == 0:
  actual.append(1)
else:
  actual.append(0)
if y == 0:
  actual.append(1)
else:
  actual.append(0)

NN = NeuralNetwork.NeuralNetwork(2, [2, 2], 2)
NN.set_input_layer([x, y])
prediction = NN.network_forward_pass()

for i in NN.input_layer:
    print(i.inputs)
    print("weights: " + str(i.weights))

for j in NN.hidden_layers:
    for i in j:
      print(i.inputs)
      print("weights: " + str(i.weights))

for i in NN.output_layer:
    print(i.inputs)
    print("weights: " + str(i.weights))
    
print("prediction: " + str(prediction))
print("actual: " + str(actual))

loss = NN.calculate_loss(prediction, actual)
print("loss:" + str(loss)'''

iterations_per_epoch = 1
epochs = 1


best_hidden_weights = []
best_output_weights = []
lowest_loss = 999999999999

for epoch in range(epochs):
    for iteration in range(iterations_per_epoch):
        print("iteration: " + str(iteration))
        NN = NeuralNetwork.NeuralNetwork(2, [4,4,4,4], 2)
        '''print("iteration: " + str(iteration))'''
        x = random.randint(0, 1)
        y = random.randint(0, 1)

        actual = []
        if x == 0:
            actual.append(1)
        else:
            actual.append(0)
        if y == 0:
            actual.append(1)
        else:
            actual.append(0)

        # print weights
        '''for layer in NN.hidden_weights:
            for weights in layer:
                print(weights)
        for weights in NN.output_weights:
            print(weights)'''

        NN.set_input_layer([x, y])
        prediction = NN.network_forward_pass()
        loss = NN.cost_function(prediction, actual)
        '''print("prediction: " + str(prediction))
        print("loss: " + str(loss))'''

        if abs(loss) < lowest_loss:
            lowest_loss = abs(loss)
            best_hidden_weights = NN.hidden_weights
            best_output_weights = NN.output_weights

NN = NeuralNetwork.NeuralNetwork(2, [4,4,4,4], 2)
if len(best_output_weights) != 0 and len(best_hidden_weights) != 0:
    NN.set_weights(best_hidden_weights, best_output_weights)
'''print("iteration: " + str(iteration))'''
x = random.randint(0, 1)
y = random.randint(0, 1)

actual = []
if x == 0:
    actual.append(1)
else:
    actual.append(0)
if y == 0:
    actual.append(1)
else:
    actual.append(0)

# print weights
'''for layer in NN.hidden_weights:
    for weights in layer:
        print(weights)
for weights in NN.output_weights:
    print(weights)'''

NN.set_input_layer([x, y])
prediction = NN.network_forward_pass()
loss = NN.cost_function(prediction, actual)
print(actual)
print(prediction)
print(loss)
x_output = NN.backpropagate_error(NN.output_layer[0], x)
print('x error: ' + str(x_output))
print('y error: ' + str(NN.backpropagate_error(NN.output_layer[1], y)))
'''print(best_hidden_weights)
print(best_output_weights)
'''






