
import random
from nndraw.linalg.vector import Vector
from nndraw.db.point_store import PointStore

def test_add_point_stores_point():
    label = 0
    p = Vector([1.0, 3.0])
    ps = PointStore()
    ps.add_point(p, label)
    points = ps.get_all()
    assert len(points) == 1
    ps.clear()

def test_get_all_returns_all_points():
    label = 0
    ps = PointStore()
    for _ in range(10):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        p = Vector([x, y])
        ps.add_point(p, label)
    points = ps.get_all()
    assert len(points) == 10
    ps.clear()

def test_find_nearest_returns_closest_points():
    label = 0
    ps = PointStore()
    for _ in range(8):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        p = Vector([x, y])
        ps.add_point(p, label)
    p1 = Vector([1.0, 3.5])
    ps.add_point(p1, label)
    p2 = Vector([1.5, 3.0])
    ps.add_point(p2, label)
    query_vector = Vector([1.0, 3.0])
    k = 3
    results = ps.find_nearest(query_vector, k)
    result_vectors = [r[0] for r in results]
    assert p1 in result_vectors
    assert p2 in result_vectors
    ps.clear()
