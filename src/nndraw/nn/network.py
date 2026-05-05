from typing import Callable

from nndraw.linalg.vector import Vector
from nndraw.nn.layer import Layer

class Network:
    """
    A fully-connected feedforward neural network composed of Layer objects.

    Layers are wired in sequence — the output of one becomes the input of the
    next. The network is defined by its layer sizes (e.g. [2, 3, 1] means 2
    inputs, one hidden layer of 3 neurons, and 1 output).

    All layers share the same activation function. Training uses gradient
    descent via backpropagation — forward pass computes a prediction, then
    the error is propagated backwards through each layer to update weights.
    """

    def __init__(
            self,
            layer_sizes: list[int],
            activation: Callable[[float], float] | None = None,
            activation_derivative: Callable[[float], float] | None = None
        ):
        layer_pairs = zip(layer_sizes, layer_sizes[1:])
        self._layers: list[Layer] = []
        for input_size, output_size in layer_pairs:
            l = Layer(
                input_size,
                output_size,
                activation,
                activation_derivative
            )
            self._layers.append(l)

    def predict(self, input: Vector) -> Vector:
        """
        Run a forward pass through all layers and return the final output.

        The input flows through each layer in sequence. The output of one layer
        becomes the input of the next. Returns the output of the last layer.
        """
        result = input
        for l in self._layers:
            result = l.predict(result)
        return result
    
    def train(
            self,
            input: Vector,
            target: Vector,
            learning_rate: float
        ) -> None:
        """
        Run one training step: forward pass, compute error, backpropagate.

        The error is the difference between the prediction and the target.
        That error signal is passed backwards through each layer in reverse
        order — each layer updates its own weights and passes the gradient
        further back.
        """
        result = self._forward(input)
        grad = Vector([o - t for o, t in zip(result, target)])
        for l in reversed(self._layers):
            grad = l.backward(grad, learning_rate)
    
    def _forward(self, input: Vector) -> Vector:
        """
        Caching forward pass used by train(). Each layer stores its input and
        pre-activation so the subsequent backward() call can compute gradients.
        Mirrors predict(), but with mutation — only call from the training thread.
        """
        result = input
        for l in self._layers:
            result = l.forward(result)
        return result
