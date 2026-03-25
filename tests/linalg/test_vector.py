from nndraw.linalg.vector import Vector

def test_vector_stores_components():
    v = Vector([1.0, 2.0, 3.0])
    assert v[0] == 1.0
    assert v[1] == 2.0
    assert v[2] == 3.0

def test_vector_length():
    v = Vector([1.0, 2.0, 3.0])
    assert len(v) == 3

def test_vector_add():
    a = Vector([1.0, 2.0, 3.0])
    b = Vector([4.0, 5.0, 6.0])
    result = a + b
    assert result[0] == 5.0
    assert result[1] == 7.0
    assert result[2] == 9.0

def test_vector_scale():
    v = Vector([1.0, 2.0, 3.0])
    result = v * 2.0
    assert result[0] == 2.0
    assert result[1] == 4.0 
    assert result[2] == 6.0

def test_vector_dot_product():
    a = Vector([1.0, 2.0, 3.0])
    b = Vector([4.0, 5.0, 6.0])
    assert a.dot(b) == 32.0