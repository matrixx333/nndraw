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
