from __future__ import annotations

class Vector:
    def __init__(self, components: list[float]):
        self._components = list(components)

    def dot(self, other: Vector) -> float:
        """
        Dot product: a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ

        Multiplies corresponding components and sums them into a single scalar.
        Geometrically, a·b = |a||b|cos(θ), so it measures how aligned two vectors
        are. Used heavily in matrix multiplication and neural network weighted sums.
        """
        pairs = zip(self._components, other._components)
        result = [a * b for a, b in pairs]
        return sum(result)

    def times(self, other: Vector) -> Vector:
        pairs = zip(self._components, other._components)
        return Vector([a * b for a, b in pairs])

    def __getitem__(self, index: int) -> float:
        return self._components[index]

    def __len__(self) -> int:
        return len(self._components)

    def __add__(self, other: Vector) -> Vector:
        """
        Vector addition: [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]

        Adds corresponding components element-wise. Geometrically, placing b's
        tail at a's tip gives the resultant vector. Used in neural networks to
        add bias terms to weighted sums.
        """
        pairs = zip(self._components, other._components)
        return Vector([a + b for a, b in pairs])

    def __mul__(self, scalar: float) -> Vector:
        """
        Scalar multiplication: [a₁·s, a₂·s, ..., aₙ·s]

        Stretches or shrinks every component by the same factor. Direction stays
        the same (or reverses if s < 0); only the magnitude changes. Used in
        backpropagation when scaling gradients by a learning rate.
        """
        return Vector([a * scalar for a in self._components])

    def __repr__(self) -> str:
        return f"Vector({self._components})"
    
    def __eq__(self, other: Vector) -> bool:
        pairs = zip(self._components, other._components)
        return all(a == b for a, b in pairs)
