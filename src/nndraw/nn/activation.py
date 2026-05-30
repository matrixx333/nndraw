from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Activation: 
    fn: Callable[[float], float]
    derivative: Callable[[float], float]