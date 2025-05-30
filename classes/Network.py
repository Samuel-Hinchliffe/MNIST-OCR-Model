from classes.Node import Node
from classes.Edge import Edge
from classes.Layer import Layer
import numpy as np
import time
import json
import matplotlib.pyplot as plt


class Network:

    def __init__(self):
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def connect_layers(self) -> list[Layer]:
        # For every given layer
        for i in range(len(self.layers) - 1):

            # And for every node in the current layer
            for from_node in self.layers[i].nodes:

                # Connect to every node in the next layer
                for to_node in self.layers[i + 1].nodes:

                    # Create an edge between the two nodes
                    edge = Edge(
                        f"{from_node.name} <---> {to_node.name}", from_node, to_node
                    )

                    # Bidirectional connection
                    from_node.outgoing.append(edge)
                    to_node.incoming.append(edge)
        return self.layers

    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: list[float]) -> float:
        # We clip the predictions to avoid log(0) error
        # Neat trick.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)

    # This will be used to 'teach' the model
    # by adjusting the weights of the edges
    # and the biases of the nodes
    # by updating the gradient of the nodes
    def back_propagate(self, y_true: np.ndarray) -> None:

        # This will be run after the feed forward pass / inference

        # Output layer (go over each node in the output layer and calculate the gradient)
        # The gradient for each output node will be: output - expected
        for i, node in enumerate(self.layers[-1].nodes):
            # If using softmax, we'll calculate the derivative for softmax + cross-entropy here
            # The gradient for each output node will be: output - expected
            node.gradient = node.value - y_true[i]

        # Propagate backwards through the network
        # Starting from the second to last layer
        # and going backwards to the first layer (looks complicated, but it's just a reverse for loop, Python...)
        for layer_idx in range(len(self.layers) - 2, -1, -1):

            layer = self.layers[layer_idx]

            for node in layer.nodes:
                # Compute the gradient for each node in the layer
                # it's defaulted to 0, but we reset it to 0 here
                node.gradient = 0

                # For each outgoing edge from the node, we will calculate the gradient
                # How much this node contributed to the output of the next layer?
                for edge in node.outgoing:
                    node.gradient += edge.weight * edge.target.gradient

                # Multiply by the derivative of the activation function
                if node.activation_name == "sigmoid":
                    node.gradient *= node.value * (1 - node.value)
                elif node.activation_name == "relu":
                    node.gradient *= 1 if node.value > 0 else 0

        # Update weights (and biases) using the gradients
        # For each layer (except the input layer)
        for layer in self.layers[1:]:

            # grab every node in the given layer
            for node in layer.nodes:

                # And update the weights of the edges
                for edge in node.incoming:

                    # Update the weight of the edge based on the gradient of the source node (we do this
                    # because the source node is the one that contributed to the value of the target node and therefore needs updating)
                    edge.weight -= 0.01 * edge.source.value * node.gradient
                    edge.source.bias -= 0.01 * node.gradient

    def load_model(self, filename: str = "model_structure.json") -> None:
        with open(filename, "r") as f:
            model_structure = json.load(f)

        # Clear existing layers
        self.layers = []

        # Recreate the layers and nodes from the model structure
        node_lookup = []  # To keep track of nodes for edge assignment
        for layer_data in model_structure["layers"]:
            activation = (
                layer_data["nodes"][0]["activation"]
                if layer_data["nodes"]
                else "sigmoid"
            )

            nodes = []
            for node_data in layer_data["nodes"]:
                node = Node(
                    name=node_data["name"],
                    activation=node_data["activation"],
                    bias=node_data["bias"],
                    value=None,
                    incoming=None,
                    outgoing=None,
                    gradient=None,
                )
                nodes.append(node)

            layer = Layer(
                layer_data["name"],
                size=len(layer_data["nodes"]),
                activation=activation,
                nodes=nodes,
            )
            self.add_layer(layer)
            node_lookup.append(nodes)

        # Reconnect the layers and set edge weights
        # ! This was nightmare fuel.
        for i in range(len(self.layers) - 1):
            from_nodes = node_lookup[i]
            to_nodes = node_lookup[i + 1]
            for to_idx, to_node in enumerate(to_nodes):
                weights = model_structure["layers"][i + 1]["nodes"][to_idx]["weights"]
                for from_idx, from_node in enumerate(from_nodes):
                    edge = Edge(
                        f"{from_node.name} <---> {to_node.name}", from_node, to_node
                    )
                    edge.weight = weights[from_idx]
                    from_node.outgoing.append(edge)
                    to_node.incoming.append(edge)

    def save_model(self, filename: str = "model_structure.json") -> None:
        # I like to save the model structure to a JSON file. Easier to visually inspect.
        # But of course, not optimal.
        model_structure: dict = {
            "layers": [
                {
                    "name": layer.name,
                    "nodes": [
                        {
                            "name": node.name,
                            "activation": node.activation_name,
                            "weights": [edge.weight for edge in node.incoming],
                            "bias": node.bias,
                        }
                        for node in layer.nodes
                    ],
                }
                for layer in self.layers
            ]
        }
        with open(filename, "w") as f:
            json.dump(model_structure, f, indent=4)

    def inference(self, image: np.ndarray) -> int:

        # Flatten the image
        image = image.flatten() / 255.0

        # Forward pass / inference
        self.feed_forward(image)

        # get output vector
        y_pred = [node.value for node in self.layers[-1].nodes]
        print_y_pred = np.round(y_pred, 2).tolist()
        print(f"Predicted: {print_y_pred}")

        # get winner / the closet node to 1
        winner_index = int(np.argmax(y_pred))

        print(f"Predicted: {winner_index} - {print_y_pred}")
        return winner_index

    def evaluate(self, test_images, test_labels):
        # Evaluate the model on the test data
        correct = 0
        total_loss = 0
        total_images = len(test_images)
        print("Evaluating model...")
        print("Total images:", total_images)

        for image, label in zip(test_images, test_labels):

            # Flatten the image
            image = image.flatten() / 255.0

            # Forward pass / inference
            self.feed_forward(image)

            # get output vector
            y_pred = [node.value for node in self.layers[-1].nodes]
            # print_y_pred = np.round(y_pred, 2).tolist()

            # get winner / the closet node to 1
            winner_index = np.argmax(y_pred)

            if winner_index == label:
                correct += 1

            running_accuracy = (correct / total_images) * 100
            print(
                f"Predicted: {winner_index}, Actual: {label} - Correct: {winner_index == label} - {running_accuracy:.2f}%"
            )

        wrong = total_images - correct
        print(
            f"Evaluation completed - Correct: {correct}, Wrong: {wrong}, "
            f"Accuracy: {(correct / total_images) * 100:.2f}% - Loss: {total_loss / total_images:.4f}"
        )

    def train(
        self,
        train_images,
        train_labels,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> None:

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            correct = 0
            num_batches = (len(train_images) + batch_size - 1) // batch_size

            print(f"\nEpoch {epoch + 1}/{epochs} starting...")

            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                i = batch_idx * batch_size

                # Get the batch of images and labels (we use slicing to get the exact batch)
                batch_images = train_images[i : i + batch_size]
                batch_labels = train_labels[i : i + batch_size]

                for image, label in zip(batch_images, batch_labels):
                    # Flatten the image
                    image = image.flatten() / 255.0  # Normalize the input

                    # Perform a forward pass
                    self.feed_forward(image)

                    # Compute the loss (Cross-Entropy)
                    # 1. Get the predicted values from the output layer
                    y_pred = [node.value for node in self.layers[-1].nodes]

                    # 2. Create the true label vector (Expected output)
                    t_true = np.zeros(10)
                    t_true[label] = 1

                    # 3. Something to read the predicted values
                    # print_y_pred = np.round(y_pred, 2).tolist()

                    # 4. Check if the prediction was correct
                    was_correct = np.argmax(y_pred) == label

                    # 5. Was correct?
                    correct += 1 if was_correct else 0

                    # How accurate so far?
                    accuracy = 0 if (i + 1) == 0 else (correct / (i + 1)) * 100

                    # get the loss
                    loss = self.cross_entropy_loss(t_true, y_pred)
                    total_loss += loss

                    # print(
                    #     f"Predicted: {print_y_pred}, Actual: {label} - Correct: {was_correct} - Loss: {loss:.4f}"
                    #     f" - Gradient: {self.layers[-1].nodes[label].gradient:.4f} - Bias: {self.layers[-1].nodes[label].bias:.4f} - Accuracy: {accuracy:.2f}%"
                    # )

                    # Perform back-propagation
                    self.back_propagate(t_true)

                batch_time = time.time() - batch_start_time
                print(
                    f"Batch {batch_idx + 1}/{num_batches} completed in {batch_time:.2f}s "
                    f"({min(i + batch_size, len(train_images))}/{len(train_images)} samples processed) ({accuracy:.2f}% accuracy)"
                )

            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {total_loss / len(train_images):.4f} - Accuracy: {accuracy:.2f}%"
            )

            # We save the model after every epoch
            self.save_model(
                filename=f"model_structure_epoch_{epoch + 1}_accuracy__{accuracy}.json"
            )

    def feed_forward(self, input_data) -> None:

        # Set the input layer values
        # populate the input layer with the input data
        for i, node in enumerate(self.layers[0].nodes):
            node.value = input_data[i]

        # Activate each layer except the input layer as we set the values above with
        # the input data
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.activate()

        # Compute softmax for the output layer
        if self.layers[-1].activation_name == "softmax":
            self.layers[-1].compute_softmax()
