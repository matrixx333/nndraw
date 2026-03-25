from nndraw.nn.activations import sigmoid, sigmoid_derivative, relu, relu_derivative

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

