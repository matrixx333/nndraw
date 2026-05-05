# Session Handoff

## Project

`nndraw` — an interactive neural network decision boundary visualizer built from scratch to learn Python, linear algebra, and Qdrant simultaneously. The user is an experienced C#/SQL developer (11yr) learning Python for the first time. Claude teaches, the user writes all code.

## Completed Steps

| Step | What was built |
|---|---|
| v1.0 MVP | linalg (Vector, Matrix), nn (Layer, Network, activations), db (PointStore), ui (Canvas) |
| v1.1 | `CanvasConfig` frozen dataclass in `src/nndraw/ui/config.py` — all canvas constants centralized |
| v1.2 | `Matrix.determinant()` and `Matrix.inverse()` added to `src/nndraw/linalg/matrix.py` |
| v1.3 | `tanh`, `tanh_derivative`, `leaky_relu`, `leaky_relu_derivative` added to `src/nndraw/nn/activations.py` |
| v1.4 | Training moved to a background thread in `canvas.py` using `threading.Lock` |
| v1.5 (in progress) | Wiring `PointStore` into `canvas.py` for persistence — partially done |

## Current State (v1.5 in progress)

`PointStore` is being integrated into `canvas.py`. Two things were done this session:

1. `add_point` is called when the user clicks to place a point
2. `get_all` loads saved points on startup — but this was blocking the main thread, causing the UI to be black on startup while Qdrant paginated through all stored points

**Last change discussed:** Move `get_all` to a background thread using a `_load_points` method:

```python
def _load_points(self) -> None:
    points = self._point_store.get_all()
    with self._lock:
        self._points = points
```

Called in `run()` as:

```python
threading.Thread(target=self._load_points, daemon=True).start()
```

This may or may not have been implemented yet — check `canvas.py` to confirm.

## What's Next

After v1.5 is confirmed working (points persist across sessions, UI loads immediately):

- **v1.6** — Multi-class classification: add more color classes, softmax output layer
- **v1.7** — LA: eigenvalues/eigenvectors, PCA overlay
- **v1.8** — Python generators & itertools, batch training with mini-batches

## Key Files

| File | Purpose |
|---|---|
| `src/nndraw/ui/canvas.py` | Main UI — Pygame loop, threading, point handling |
| `src/nndraw/ui/config.py` | `CanvasConfig` frozen dataclass |
| `src/nndraw/db/point_store.py` | Qdrant repository — `add_point`, `get_all`, `find_nearest` |
| `src/nndraw/linalg/matrix.py` | Matrix with `determinant()` and `inverse()` |
| `src/nndraw/nn/activations.py` | sigmoid, relu, tanh, leaky_relu + derivatives |
| `docs/plan.md` | Full project roadmap |

## Current Config Values

```python
width: int = 1600
height: int = 900
fps: int = 60
grid_size: int = 25
learning_rate: float = 0.05
hidden_size: int = 6
```

## Teaching Notes

- User writes all code — Claude explains concepts first, then assigns the task, waits for the user to write it
- TDD throughout — tests written before implementation
- User is comfortable with Python now, explain concepts but don't over-explain basics
- User prefers to move at a good pace — don't dwell on optional improvements unless they ask
- When the user asks Claude to write something small (docstrings, minor edits), do it — don't always push back
