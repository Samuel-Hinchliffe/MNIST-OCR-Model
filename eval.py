from classes.Network import Network
import time
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import time

# Quick script to evaluate the model post training.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = Network()
network.load_model("./model/model_structure_epoch_5_accuracy__95.json")

print("Starting evaluation...")
start_time = time.time()

# You will see the results from within this function
network.evaluate(test_images, test_labels)

end_time = time.time()
print(f"Time taken to train: {end_time - start_time:.2f} seconds")
