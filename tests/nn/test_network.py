from nndraw.linalg.vector import Vector
from nndraw.nn.network import Network
from nndraw.nn.activations import sigmoid, sigmoid_derivative

layer_sizes = [2, 3, 1]

def test_network_predict_identity():    
    input = Vector([5.0, 5.0])
    n = Network(layer_sizes)
    output = n.predict(input)
    assert len(output) == layer_sizes[-1]

def test_network_train_reduces_loss():
    learning_rate = 1.0
    input = Vector([0.5, 0.5])    
    target = Vector([1.0])
    n = Network(layer_sizes, sigmoid, sigmoid_derivative)
    initial_output = n.predict(input)
    for i in range(10000):
        n.train(input, target, learning_rate)
    final_output = n.predict(input)
    initial_sum = mean_squared_error(initial_output, target)
    final_sum = mean_squared_error(final_output, target)    
    assert final_sum < initial_sum

def mean_squared_error(output: Vector, target: Vector) -> float:
    return sum((o - t) ** 2 for o, t in zip(output, target))