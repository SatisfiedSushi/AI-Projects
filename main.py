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
prediction = NN.network_forward_pass()'''

'''for i in NN.input_layer:
    print(i.inputs)
    print("weights: " + str(i.weights))

for j in NN.hidden_layers:
    for i in j:
      print(i.inputs)
      print("weights: " + str(i.weights))

for i in NN.output_layer:
    print(i.inputs)
    print("weights: " + str(i.weights))'''
    
'''print("prediction: " + str(prediction))
print("actual: " + str(actual))

loss = NN.calculate_loss(prediction, actual)
print("loss:" + str(loss))'''

iterations_per_epoch = 1000
epochs = 10

for epoch in range(epochs):
  for iteration in range(iterations_per_epoch):









