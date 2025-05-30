import math
import random
from typing import List, Optional
from classes.Edge import Edge
from classes.Layer import Layer
from classes.Network import Network


class Node:
    def __init__(
        self,
        name: Optional[str] = None,
        activation: str = "sigmoid",
        bias: Optional[float] = None,
        incoming: Optional[List["Edge"]] = None,
        outgoing: Optional[List["Edge"]] = None,
        gradient: Optional[float] = None,
        value: Optional[float] = None,
    ):

        # Just a name. Useful for debugging
        self.name = name

        # Choose the activation function.
        self.activation_name = activation
        self.activation_fn: function = self._get_activation_fn(activation)

        # All nodes have a value, which is the output of the node.
        self.value = 0.0 if value is None else value

        # A bias is a constant added to the weighted sum of inputs.
        # It allows the model to fit the data better.
        self.bias = random.uniform(-1, 1) if bias is None else bias

        # Incoming edges are edges that point TO this node.
        # Outgoing edges are edges that point FROM this node.
        self.incoming = [] if incoming is None else incoming
        self.outgoing = [] if outgoing is None else outgoing

        # Used for training
        self.gradient = 0.0 if gradient is None else gradient

    def activate(self) -> None:
        # Only compute if there are incoming edges (not an input node)
        if not self.incoming:
            return

        # Compute the weighted sum of inputs
        total_input = 0
        for edge in self.incoming:
            contribution = edge.weight * edge.source.value
            total_input += contribution

        total_input += self.bias

        if self.activation_name != "softmax":
            # do nothing, the layer will handle the softmax
            # and the output will be a list of probabilities
            self.value = self.activation_fn(total_input)
        else:
            # for softmax, we need the raw logits
            # and the layer will handle the softmax
            self.value = total_input
            pass

    def _get_activation_fn(self, name: str) -> function:
        # Quick way to dynamically get the activation function
        # based on the name.
        activation_functions = {
            "sigmoid": self.sigmoid,
            "relu": self.reLU,
            "tanh": self.tanh,
        }
        if name in activation_functions:
            return activation_functions[name]
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def __str__(self) -> str:
        return f"Node(name={self.name}, value={self.value:.4f}, bias={self.bias:.4f}, activation={self.activation_name})"

    @staticmethod
    def sigmoid(x) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def reLU(x) -> float:
        return max(0, x)

    @staticmethod
    def tanh(x) -> float:
        return math.tanh(x)
