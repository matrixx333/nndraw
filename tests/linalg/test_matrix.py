from nndraw.linalg.matrix import Matrix
from nndraw.linalg.vector import Vector

def test_matrix_scores_rows():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m[0][0] == 1.0
    assert m[1][1] == 4.0

def test_matrix_shape():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.rows == 2
    assert m.cols == 2

def test_matrix_transpose():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = m.transpose()
    assert result.rows == 3
    assert result.cols == 2
    assert result[0][0] == 1
    assert result[2][1] == 6

def test_matrix_multiply_vector():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([1.0, 1.0])
    result = m * v
    assert result[0] == 3.0
    assert result[1] == 7.0

def test_matrix_multiplication():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = m1.multiply(m2)
    assert result[0][0] == 19.0
    assert result[0][1] == 22.0
    assert result[1][0] == 43.0
    assert result[1][1] == 50.0