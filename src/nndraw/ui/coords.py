from nndraw.linalg.vector import Vector
from nndraw.ui.config import CanvasConfig

_config = CanvasConfig()

def normalize(x: int, y: int) -> list[float]:
    nx = x / _config.width
    ny = y / _config.height
    return [nx, ny]

def denormalize(input: Vector) -> tuple[float, float]:
    x = input[0] * _config.width
    y = input[1] * _config.height
    return (x, y)