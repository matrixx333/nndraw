from __future__ import annotations
from typing import overload
from nndraw.linalg.vector import Vector

class Matrix:
    """
    A 2D grid of floats representing a mathematical matrix.

    Supports matrix-matrix multiplication, matrix-vector multiplication, scalar
    scaling, subtraction, and transposition. Used in the neural network to hold
    weight matrices and perform the core linear algebra of forward and backward passes.
    """

    def __init__(self, rows: list[list[float]]):
        if len(set(len(row) for row in rows)) > 1:
            raise ValueError("All rows must have the same number of columns")
        self._rows = rows

    def transpose(self) -> Matrix:
        """
        Transpose: flip the matrix over its diagonal so rows become columns.

        An m×n matrix becomes n×m, where element [i][j] moves to [j][i].
        Used when aligning dimensions for matrix multiplication, and in
        backpropagation to route gradients backwards through a layer (Wᵀ·δ).
        """
        return Matrix([[self._rows[row][col] for row in range(self.rows)] for col in range(self.cols)])
    
    def determinant(self):
        are_equal = self.cols == self.rows;
        if not are_equal:
            raise ValueError("Cannot find the determinate of a non-square matrix.")
        if not self.cols == 2:
                raise NotImplementedError()
        (a, b), (c, d) = self._rows
        return (a * d) - (b * c)
    
    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("A singular matrix has no inverse")
        (a, b), (c, d) = self._rows
        m = Matrix([[d, -b], [-c, a]])
        return (1 / self.determinant()) * m

    @property
    def rows(self) -> int:
        """Gets the rows for the Matrix"""
        return len(self._rows)

    @property
    def cols(self) -> int:
        """Gets the columns for the Matrix"""
        return len(self._rows[0])

    @overload
    def __getitem__(self, index: int) -> list[float]: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> float: ...

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return self._rows[row][col]
        return self._rows[index]

    def __sub__(self, other: Matrix) -> Matrix:
        """Element-wise subtraction. Used in gradient descent to subtract scaled weight gradients from the weight matrix."""
        pairs = zip(self._rows, other._rows)
        return Matrix([[x - y for x, y in zip(a, b)] for a, b in pairs])

    @overload
    def __mul__(self, other: Matrix) -> Matrix: ...

    @overload
    def __mul__(self, other: Vector) -> Vector: ...

    @overload
    def __mul__(self, other: float) -> Matrix: ...

    def __mul__(self, other):
        """
        Matrix multiplication, matrix-vector multiplication, or scalar scaling.

        Matrix * Matrix → standard matrix multiplication (dot products of rows and columns).
        Matrix * Vector → transforms the vector into a new space (the core of a forward pass).
        Matrix * float  → scales every element by the scalar.
        """
        if isinstance(other, Matrix):
            other_t = other.transpose()
            result = [
                [Vector(row).dot(Vector(other_t[col_i])) for col_i in range(other.cols)] for row in self._rows
            ]
            return Matrix(result)
        elif isinstance(other, Vector):
            return Vector([Vector(row).dot(other) for row in self._rows])
        else: 
            return Matrix([[cell * other for cell in row] for row in self._rows])

    def __rmul__(self, other: float) -> Matrix:
        """Allows float * Matrix in addition to Matrix * float."""
        return self.__mul__(other);

    def __repr__(self) -> str:
        return f"Matrix({self._rows})"
    
    def __eq__(self, other: Matrix) -> bool:
        pairs = zip(self._rows, other._rows)
        return all(a == b for row_a, row_b in pairs for a, b in zip(row_a, row_b))


