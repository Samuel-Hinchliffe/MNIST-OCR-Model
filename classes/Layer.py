from classes.Node import Node
import numpy as np
import types


# NN Layer class
class Layer:
    def __init__(
        self,
        name: str,
        size: int,
        activation: str = "sigmoid",
        nodes: list[Node] | None = None,
    ) -> None:
        # Just a name. For debugging purposes.
        self.name = name

        # Activation function name
        self.activation_name = activation

        # Something to hold all of our nodes
        self.nodes = (
            [Node(name=f"{name}_Node_{i}", activation=activation) for i in range(size)]
            if nodes is None
            else nodes
        )

    # Our final layer will be a set of probabilities.
    def compute_softmax(self):
        outputs = np.array([node.value for node in self.nodes])
        exp_values = np.exp(outputs - np.max(outputs))
        softmax_outputs = exp_values / np.sum(exp_values)

        for node, softmax_val in zip(self.nodes, softmax_outputs):
            node.value = softmax_val
        return softmax_outputs
