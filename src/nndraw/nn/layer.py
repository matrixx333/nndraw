import random
from typing import Callable
from nndraw.linalg.vector import Vector
from nndraw.linalg.matrix import Matrix

class Layer:
    """
    A single fully-connected layer in a neural network.

    Each neuron in this layer is connected to every input. The layer holds a
    weight matrix and a bias vector, and optionally applies an activation
    function to introduce non-linearity.

    During training, the layer stores intermediate values from the forward pass
    so that backward() can compute gradients and update the weights in-place.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: Callable[[float], float] | None = None,
            activation_derivative: Callable[[float], float] | None = None
        ):
        self._activation = activation
        self._activation_derivative = activation_derivative
        self.weights = Matrix([[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)])
        self.bias = Vector([0.0 for _ in range(output_size)])
        self._input = Vector([])
        self._last_z = Vector([])
        self._grad_bias = Vector([])
        self._grad_weights = Matrix([])

    def forward(self, input: Vector) -> Vector:
        """
        Run the input through this layer and return the output.

        Computes z = weights * input + bias, then applies the activation
        function if one was provided. Stores input and z for use in backward().
        """
        self._input = input
        self._last_z = (self.weights * input) + self.bias
        if self._activation:
            return Vector([self._activation(x) for x in self._last_z])
        return self._last_z

    def backward(self, grad_output: Vector, learning_rate: float) -> Vector:
        """
        Backpropagate the error signal and update weights and biases.

        Receives grad_output — the gradient of the loss with respect to this
        layer's output — and uses it to compute how much each weight and bias
        contributed to the error. Applies the gradient descent update rule:

            weight = weight - learning_rate * gradient

        Returns grad_input, the gradient with respect to this layer's input,
        which is passed back to the previous layer to continue backpropagation.
        """
        if self._activation_derivative:
            derivative_result = Vector([self._activation_derivative(x) for x in self._last_z])
            self._grad_bias = grad_output * derivative_result
        else:
            self._grad_bias = grad_output
        self._grad_weights = Matrix([[i * j for j in self._input] for i in self._grad_bias])
        grad_input = self.weights.transpose() * self._grad_bias
        self.weights = self.weights - (self._grad_weights * learning_rate)
        self.bias = self.bias - (self._grad_bias * learning_rate)
        return grad_input