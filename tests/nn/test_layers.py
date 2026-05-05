from typing import Callable

from nndraw.linalg.matrix import Matrix
from nndraw.linalg.vector import Vector
from nndraw.nn.layer import Layer
from nndraw.nn.activations import sigmoid

learning_rate = 1.0

def test_layer_initializes_correct_shape():
    l = Layer(input_size=2, output_size=3)
    assert l.weights.rows == 3
    assert l.weights.cols == 2
    assert len(l.bias) == 3

def test_layer_forward_identity():
    l = _get_layer()
    input = Vector([3.0, 5.0])
    output = l.forward(input)
    assert output == input

def test_layer_forward_applies_activation():
    l = _get_layer(sigmoid)
    input = Vector([5.0, 0.0])
    output = l.forward(input)
    assert output[0] == 0.9933071490757153
    assert output[1] == 0.5

def test_layer_backward_returns_grad_input():
    l = _get_layer()
    input = Vector([1.0, 1.0])
    grad_output = Vector([1.0, 1.0])
    l.forward(input)
    grad_input = l.backward(grad_output, learning_rate)
    assert grad_output == grad_input

def test_layer_backward_stores_grad_weights():
    l = _get_layer()
    input = Vector([5.0, 5.0])
    grad_output = Vector([1.0, 1.0])
    l.forward(input)
    _ = l.backward(grad_output, learning_rate)
    assert l._grad_weights == Matrix([[5.0, 5.0], [5.0, 5.0]])

def test_layer_backward_stores_grad_bias():
    l = _get_layer()
    input = Vector([1.0, 1.0])
    grad_output = Vector([5.0, 5.0])
    l.forward(input)
    _ = l.backward(grad_output, learning_rate)
    assert l._grad_bias == Vector([5.0, 5.0])

def test_predict_doesnt_mutate_input_and_last_z():
    l = _get_layer()
    l.predict(Vector([3.0, 5.0]))
    l.predict(Vector([1.0, 2.0]))
    assert l._input is None
    assert l._last_z is None

def _get_layer(activation: Callable[[float], float] | None = None) -> Layer:
    l = Layer(input_size=2, output_size=2, activation=activation)
    l.weights = Matrix([[1.0, 0.0], [0.0, 1.0]])
    l.bias = Vector([0.0, 0.0])
    return l