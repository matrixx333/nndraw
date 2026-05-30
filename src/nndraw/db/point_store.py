from typing import cast
from nndraw.linalg.vector import Vector
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    Record, 
    PointStruct 
)

class PointStore:
    def __init__(self):
        self._collection_name = "points"
        client = QdrantClient(host="localhost", port=6333)
        self._client = client
        self._id = 0

        _create_collection(self)

    def clear(self):
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as e:
            print(f"error deleting collection: {e}")
        _create_collection(self)
    
    def add_point(self, p: Vector, label: int):
        self._id += 1
        payload = {
            "label": label
        }
        point_struct = PointStruct(
            id=self._id,
            payload=payload,
            vector=list(p)
        )
        points = [point_struct]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points
        )
    
    def get_all(self) -> list[tuple[Vector, int]]: 
        all_records: list[Record] = []
        all_vectors: list[Vector] = []
        scroll_offset = None

        while True:
            points, offset = self._client.scroll(
                  collection_name=self._collection_name,
                  with_payload=True,
                  with_vectors=True,
                  limit=100,
                  offset=scroll_offset
            )
            all_records.extend(points)

            if offset is None: 
                break

            scroll_offset = offset

        for r in all_records:
            label = r.payload['label']
            v = Vector(cast(list[float], r.vector))
            all_vectors.append((v, label))

        return all_vectors
    
    def find_nearest(
            self, 
            query_vector: Vector, 
            k: int
        ) -> list[tuple[Vector, int]]:
        results: list[tuple[Vector, int]] = []

        search_result = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            limit=k,
            with_payload=True,
            with_vectors=True
        )

        for r in search_result.points:
            if r.payload is None:
                raise ValueError(f'point {r.id} has no payload')
            label = r.payload["label"]
            v = Vector(cast(list[float], r.vector))
            results.append((v, label))

        return results
    
def _create_collection(self):
    vector_params = VectorParams(size=2, distance=Distance.EUCLID)
    if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=vector_params
            )