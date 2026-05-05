import random
from typing import Callable
from nndraw.linalg.vector import Vector
from nndraw.linalg.matrix import Matrix

class Layer:
    """
    A single fully-connected layer in a neural network.

    Each neuron in this layer is connected to every input — represented as a
    row in the weight matrix, where each element is the connection strength to
    one input. The layer holds a weight matrix and a bias vector, and optionally
    applies an activation function to introduce non-linearity.

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
        """
        activation_derivative must accept the pre-activation value z (the raw
        weighted sum W·x + b), not the post-activation value a = activation(z).
        For sigmoid, write it as σ(z)·(1 − σ(z)), not as a·(1 − a).
        """
        self._activation = activation
        self._activation_derivative = activation_derivative
        self.weights = Matrix([[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)])
        self.bias = Vector([0.0 for _ in range(output_size)])
        self._input: Vector | None = None
        self._last_z: Vector | None = None
        self._grad_bias: Vector | None = None
        self._grad_weights: Matrix | None = None

    def predict(self, input: Vector) -> Vector:
        """
        Pure forward pass — computes the layer's output without storing
        intermediate state. Used for inference (e.g. rendering the decision
        boundary) so it is safe to call concurrently with training.
        """
        return self._activate(self._compute(input))

    def forward(self, input: Vector) -> Vector:
        """
        Run the input through this layer and return the output.

        Computes z = weights * input + bias, then applies the activation
        function if one was provided. Stores input and z for use in backward().
        """
        self._input = input
        self._last_z = self._compute(input)
        return self._activate(self._last_z)

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
    
    def _compute(self, input: Vector) -> Vector:
        """
        Compute the pre-activation z = W·x + b. Returns the raw weighted sum
        without applying the activation function — callers apply activation
        themselves so backward() can rely on _last_z holding the true z value
        that activation_derivative expects.
        """
        return (self.weights * input) + self.bias

    def _activate(self, z: Vector) -> Vector:
        """Apply the activation function element-wise, or pass z through unchanged if no activation is set."""
        if self._activation:
            return Vector([self._activation(x) for x in z])
        return z