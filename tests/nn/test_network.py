from nndraw.linalg.matrix import Matrix
from nndraw.linalg.vector import Vector
from nndraw.nn.layer import Layer
from nndraw.nn.network import Network

def test_network_predict_identity():
    layers: list[Layer] = []
    identity = Matrix([[1.0, 0.0], [0.0, 1.0]])
    for _ in range(3):
        l = Layer(input_size=2, output_size=2)
        l.weights = identity
        l.bias = Vector([0.0, 0.0])
        layers.append(l)
    input = Vector([5.0, 5.0])
    layer_sizes = [2, 3, 1]
    n = Network(layer_sizes)
    n._layers = layers
    result = n.predict(input)
    assert result == input


