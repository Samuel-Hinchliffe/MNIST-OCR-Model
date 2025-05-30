import random
from typing import Any, Optional


class Edge:
    def __init__(
        self,
        name: str,
        source: Any,
        target: Any,
        weight: Optional[float] = None,
    ):
        """
        This connects two nodes together and has a weight associated with it.
        """

        # Just a name.
        self.name = name

        # Where the edge starts
        self.source = source

        # Where the edge ends
        self.target = target

        # The weight of the edge, initialized to a small random value
        # This weight will be adjusted during training
        self.weight: float = weight if weight is not None else random.uniform(-1, 1)
        self.gradient: float = 0.0

    def __repr__(self):
        # Printing in a way, that you can read :D
        return f"Edge({self.source}, {self.target}, weight={self.weight}, gradient={self.gradient})"
