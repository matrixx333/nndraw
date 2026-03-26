import random
from typing import Callable
from nndraw.linalg.vector import Vector
from nndraw.linalg.matrix import Matrix

class Layer:
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
        self._input = input
        self._last_z = (self.weights * input) + self.bias
        if self._activation:
            return Vector([self._activation(x) for x in self._last_z])
        return self._last_z

    def backward(self, grad_output: Vector) -> Vector:
        if self._activation_derivative:
            derivative_result = Vector([self._activation_derivative(x) for x in self._last_z])
            self._grad_bias = grad_output.times(derivative_result)
        else:
            self._grad_bias = grad_output
        self._grad_weights = Matrix([[i * j for j in self._input] for i in self._grad_bias])
        grad_input = self.weights.transpose() * self._grad_bias
        return grad_input