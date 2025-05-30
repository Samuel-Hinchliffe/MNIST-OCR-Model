from classes.Layer import Layer
from classes.Network import Network
from classes.Node import Node
from classes.Edge import Edge
import random
import math
import time
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import json
import numpy as np
import time


# Let's create a simple neural network with 3 layers
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = Network()

# Input Layer (784 Nodes, Sigmoid Activation)
input_layer = Layer("Input Layer", size=784, activation="sigmoid")

# Hidden Layer (128 Nodes, ReLU Activation)
hidden_layer = Layer("Hidden Layer", size=64, activation="relu")

# Output Layer (10 Nodes, Softmax Activation)
output_layer = Layer("Output Layer", size=10, activation="softmax")

# Add layers to the network
network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(output_layer)
network.connect_layers()

print("Starting training...")
start_time = time.time()

# 4 epochs seems to be a good number for this dataset from a few tests
network.train(train_images, train_labels, epochs=4, batch_size=32)
end_time = time.time()
print(f"Time taken to train: {end_time - start_time:.2f} seconds")
