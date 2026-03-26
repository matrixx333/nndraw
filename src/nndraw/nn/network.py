from nndraw.linalg.matrix import Matrix
from nndraw.linalg.vector import Vector
from nndraw.nn.layer import Layer

class Network: 
    def __init__(self, layer_sizes: list[int]):
        layer_pairs = zip(layer_sizes, layer_sizes[1:])
        self._layers: list[Layer] = []
        for input_size, output_size in layer_pairs:
            self._layers.append(Layer(input_size=input_size, output_size=output_size))

    def predict(self, input: Vector) -> Vector: 
        result = input
        for l in self._layers:
            result = l.forward(result)
        return result