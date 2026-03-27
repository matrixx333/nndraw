from __future__ import annotations

class Vector:
    """
    An ordered list of floats representing a mathematical vector.

    Supports element-wise addition, subtraction, and multiplication, scalar
    scaling, dot product, and iteration. Used throughout the neural network as
    inputs, outputs, biases, and gradient signals.
    """

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
    
    def __sub__(self, other: Vector) -> Vector:
        """Element-wise subtraction. Used in gradient descent to subtract scaled gradients from weights."""
        pairs = zip(self._components, other._components)
        return Vector([a - b for a, b in pairs])

    def __mul__(self, other) -> Vector:
        """
        Element-wise multiplication with another Vector, or scalar scaling with a float.

        Vector * Vector → each component multiplied by its counterpart.
        Vector * float  → every component scaled by the scalar.
        Used in backpropagation to apply the activation derivative element-wise.
        """
        if isinstance(other, Vector):
            pairs = zip(self._components, other._components)
            return Vector([a * b for a, b in pairs])
        else: 
            return Vector([a * other for a in self._components])

    def __rmul__(self, other: float) -> Vector:
        """Allows float * Vector in addition to Vector * float."""
        return self.__mul__(other);

    def __repr__(self) -> str:
        return f"Vector({self._components})"
    
    def __eq__(self, other: Vector) -> bool:
        pairs = zip(self._components, other._components)
        return all(a == b for a, b in pairs)

    def __iter__(self):
        return iter(self._components)