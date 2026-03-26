# Project Plan: Interactive Neural Network Visualizer

## Context

An experienced C#/SQL developer (11yr) is learning Python, linear algebra, a modern vector database, **and machine learning fundamentals** simultaneously. They want a guided, TDD-based experience where they write the code and Claude teaches — **no prior ML knowledge is assumed**. The chosen project is an **interactive neural network decision boundary visualizer** — the user clicks to place colored data points on a 2D canvas, a neural network (built from scratch using only custom linear algebra) trains on those points in real time, and the canvas background colors shift to show the evolving decision boundary.

**Teaching approach:** Each step introduces new concepts with clear explanations — what it is, why it matters, and how it connects to what was already built. ML concepts (neurons, weights, loss, gradient descent, backpropagation) are introduced gradually as needed, never assumed.

**What makes this ideal:**
- Linear algebra is *essential*, not cosmetic (matrix multiply = forward pass, gradient vectors = backprop)
- ML concepts emerge naturally from the code — no black-box libraries hiding the mechanics
- Qdrant vector DB fits naturally (training points stored as 2D vectors; ANN queries for nearest-neighbor comparison)
- Pygame makes it fully interactive and visual
- No ML libraries — the user implements every neuron themselves, so they truly understand it
- TDD is natural — each layer can be tested in isolation before wiring up

---

## Subject

**Interactive 2D Decision Boundary Visualizer**

1. User clicks on a Pygame canvas to place pastel green or purple data points
2. A neural network trains continuously on those points
3. The canvas background is colored (pastel green/purple gradient) based on the network's current predictions across a grid
4. User watches the decision boundary morph in real time as training progresses
5. Controls: toggle class (green/purple), reset, pause/resume training, adjust learning rate

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13.5 |
| Linear algebra | Custom library (no third-party) |
| Visualization | Pygame |
| Database | Qdrant (local, via Docker or embedded) |
| DB client | `qdrant-client` |
| Testing | pytest |
| Project config | pyproject.toml (PEP 517/518) |

**Why Qdrant:** Training points are stored as 2D vectors. Qdrant's ANN (approximate nearest-neighbor) search answers "what training points are closest to this grid cell?" — used to show a KNN overlay alongside the neural network prediction for comparison. This makes the DB a meaningful part of the app, not a bolted-on afterthought.

**Qdrant setup:** Docker container (`docker run -p 6333:6333 qdrant/qdrant`). Client connects to `localhost:6333`.

---

## Project Structure

```
python-project/
├── src/
│   └── nndraw/
│       ├── __init__.py
│       ├── linalg/           # Custom linear algebra (no third-party)
│       │   ├── __init__.py
│       │   ├── vector.py     # Vector class
│       │   └── matrix.py     # Matrix class
│       ├── nn/               # Neural network (uses only linalg/)
│       │   ├── __init__.py
│       │   ├── activations.py
│       │   ├── layer.py
│       │   └── network.py
│       ├── db/               # Qdrant repository
│       │   ├── __init__.py
│       │   └── point_store.py
│       └── ui/              # Pygame visualization
│           ├── __init__.py
│           └── canvas.py
├── tests/
│   ├── linalg/
│   ├── nn/
│   └── db/
├── pyproject.toml
├── CLAUDE.md                 # Update after setup
└── docs/
    └── goals.md
```

---

## Build Order (TDD throughout)

### Step 1 — Project scaffold
- Create `pyproject.toml` with pytest config and dependencies
- Create package skeleton (`__init__.py` files, empty modules)
- Verify: `python -m pytest` runs (zero tests pass, zero fail)

### Step 2 — `linalg/vector.py`
Teach: Python classes, `__init__`, `__repr__`, operator overloading (`__add__`, `__mul__`, etc.)
Implement (test-first): `Vector(components)`, `add`, `sub`, `scale`, `dot`, `magnitude`, `normalize`

### Step 3 — `linalg/matrix.py`
Teach: list comprehensions, nested loops, Python type hints
Implement (test-first): `Matrix(rows)`, `multiply(matrix)`, `multiply_vector(v)`, `transpose()`, element access

### Step 4 — `nn/activations.py`
Teach: pure functions, math module
Implement (test-first): `sigmoid(x)`, `sigmoid_derivative(x)`, `relu(x)`, `relu_derivative(x)`

### Step 5 — `nn/layer.py`
Teach: encapsulation, random weight init, forward pass
Implement (test-first): `Layer(input_size, output_size)` with weight matrix + bias vector, `forward(inputs) -> outputs`, `backward(...)` returning gradients

### Step 6 — `nn/network.py`
Teach: composing objects, training loop
Implement (test-first): `Network(layer_sizes)`, `predict(inputs)`, `train(inputs, targets, lr)`

### Step 7 — `db/point_store.py`
Teach: Qdrant Python client, collections, upsert, search
Implement: `PointStore`, `add_point(x, y, label)`, `get_all()`, `find_nearest(x, y, k)`

### Step 8 — `ui/canvas.py`
Teach: Pygame event loop, surface drawing, threading for background training
Implement: interactive canvas that ties all layers together

---

## Expansion Roadmap (future learning milestones)

This project is designed to grow. After the MVP, each expansion introduces new Python, LA, or DB concepts:

| Phase | New concept | What gets added |
|---|---|---|
| MVP | Basics | 2-layer network, 2D canvas, Qdrant point storage |
| v1.1 | Python dataclasses & enums | Refactor config and hyperparameters |
| v1.2 | LA: matrix inverse & determinants | Add analytical weight solutions, visualize matrix properties |
| v1.3 | More activation functions | Tanh, Leaky ReLU — compare convergence visually |
| v1.4 | Python async/threading | Background training thread, smooth animation |
| v1.5 | Qdrant snapshots & persistence | Save/load sessions; Qdrant collections on disk |
| v1.6 | Multi-class classification | Add more color classes; softmax output layer |
| v1.7 | LA: eigenvalues/eigenvectors | PCA overlay to visualize feature directions |
| v1.8 | Python generators & itertools | Batch training with mini-batches |
| v1.9 | Qdrant filtering & payloads | Rich metadata on points; filter by training epoch |
| v2.0 | Convolutional concepts | Extend to image patch classification |
| v2.1 | Real datasets | Swap mouse-drawn points for real data; introduce CSV/data pipeline concepts |
| v2.2 | Genomics domain | Classify gene sequences or CRISPR guide RNA efficiency; DNA/RNA features as vectors — same network, new data source |
| v2.3 | Python data pipeline | `csv` / `json` module, data normalization, train/test split — proper ML workflow |
| v2.4 | Qdrant as data warehouse | Persist domain datasets in Qdrant collections; switch datasets at runtime |

---

## Critical Files to Create/Modify

- `f:\code\python-project\pyproject.toml` — create
- `f:\code\python-project\CLAUDE.md` — update with commands
- `f:\code\python-project\src\nndraw\` — create (full tree)
- `f:\code\python-project\tests\` — create (full tree)

---

## Verification

1. `python -m pytest` — all unit tests pass
2. `python -m nndraw` — Pygame window opens
3. Click to place pastel green and purple points → background colors shift as network trains
4. Qdrant collection visible via Qdrant dashboard at `http://localhost:6333/dashboard` after placing points
