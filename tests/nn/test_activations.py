import pytest

from nndraw.nn.activations import (
    sigmoid, 
    sigmoid_derivative, 
    relu, 
    relu_derivative, 
    tanh, 
    tanh_derivative,
    leaky_relu,
    leaky_relu_derivative
)


def test_sigmoid_returns_zero_point_five_at_zero():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.5
    assert sigmoid(-100) < 0.5

def test_sigmoid_derivative_at_zero():
    assert sigmoid_derivative(0) == 0.25

def test_relu_clamps_negative_to_zero():
    assert relu(-100) == 0

def test_relu_returns_positive_unchanged():
    assert relu(100) == 100

def test_relu_derivative_claps_negative_to_zero():
    assert relu_derivative(-100) == 0

def test_relu_derivative_returns_one_for_positive_number():
    assert relu_derivative(100) == 1

def test_tanh():
    assert tanh(2) == pytest.approx(0.9640275801)

def test_tanh_derivative():
    t_d = tanh_derivative(2)
    assert t_d == pytest.approx(0.0706508249)

def test_leaky_relu_larger_than_zero():
    x = 2
    result = leaky_relu(x)
    assert result == x

def test_leaky_relu_at_zero():
    x = -1
    result = leaky_relu(x)
    assert result == -0.01

def test_leaky_relu_derivative_larger_than_zero():
    x = 2
    result = leaky_relu_derivative(x)
    assert result == 1

def test_leaky_relu_derivative_less_than_zero():
    x = -1
    result = leaky_relu_derivative(x)
    assert result == 0.01
