from nndraw.linalg.vector import Vector

class Matrix:
    def __init__(self, rows: list[list[float]]):
        if len(set(len(row) for row in rows)) > 1:
            raise ValueError("All rows mut have the same number of columns")
        self._rows = rows

    def transpose(self) -> "Matrix":
        """
        Transpose: flip the matrix over its diagonal so rows become columns.

        An m×n matrix becomes n×m, where element [i][j] moves to [j][i].
        Used when aligning dimensions for matrix multiplication, and in
        backpropagation to route gradients backwards through a layer (Wᵀ·δ).
        """
        return Matrix([[self._rows[row][col] for row in range(self.rows)] for col in range(self.cols)])

    def multiply(self, other: "Matrix") -> "Matrix":
        """
        Matrix multiplication: C[i][j] = dot(row i of self, column j of other)

        Requires self.cols == other.rows and produces a (self.rows × other.cols)
        matrix. We transpose `other` so its columns become rows, letting us
        reuse the Vector dot product for each cell.
        """
        other_t = other.transpose()
        result = [
            [Vector(row).dot(Vector(other_t[col_i])) for col_i in range(other.cols)] for row in self._rows
        ]
        return Matrix(result)

    @property
    def rows(self) -> int:
        return len(self._rows)

    @property
    def cols(self) -> int:
        return len(self._rows[0])

    def __getitem__(self, index: int) -> list[float]:
        if isinstance(index, tuple):
            row, col = index
            return self._rows[row][col]
        return self._rows[index]

    def __mul__(self, vector: "Vector") -> "Vector":
        """
        Matrix-vector multiplication: output[i] = dot(row i of self, vector)

        An m×n matrix applied to an n-dimensional vector produces an m-dimensional
        vector. This is the core operation of a neural network layer — output = W·x —
        where each row of W computes one neuron's weighted sum over the inputs.
        """
        return Vector([Vector(row).dot(vector) for row in self._rows])

    def __repr__(self) -> str:
        return f"Matrix({self._rows})"
