class Vector:
    def __init__(self, components: list[float]):
        self._components = list(components)

    def __getitem__(self, index: int) -> float:
        return self._components[index]

    def __len__(self) -> int:
        return len(self._components)
    
    def __add__(self, other: "Vector") -> "Vector":
        pairs = zip(self._components, other._components)
        return Vector([a + b for a, b in pairs])
    
    def __mul__(self, scalar: float) -> "Vector":
        return Vector([a * scalar for a in self._components])
    
    def dot(self, other: "Vector") -> float:
        pairs = zip(self._components, other._components)
        result = [a * b for a, b in pairs]
        return sum(result)
