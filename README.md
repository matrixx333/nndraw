# nndraw

An interactive neural network decision boundary visualizer — built from scratch to learn Python, linear algebra, and vector databases hands-on.

This project intentionally avoids third-party math libraries. Every vector operation, matrix multiplication, and dot product is hand-rolled to build real intuition for the linear algebra powering neural networks. The UI renders in real time using Pygame, and Qdrant stores the training points so you can query and explore them as vectors.

<img width="640" height="373" alt="nndraw" src="https://github.com/user-attachments/assets/2db8b066-8a7b-4b24-bf7e-82fbdc894183" />


## What's inside

- **`linalg/`** — A custom linear algebra library: vectors, matrices, and the math primitives that underpin everything else. No NumPy, no SciPy.
- **`nn/`** — A neural network built on top of `linalg/`, capable of learning a decision boundary you can watch update in real time.
- **`db/`** — A Qdrant repository layer that persists training data as vector embeddings and supports similarity search.
- **`ui/`** — An interactive Pygame canvas where you place training points by clicking, trigger training, and see the decision boundary evolve.

## Prerequisites

- Python 3.13+
- [Docker](https://www.docker.com/) (for Qdrant)

## Getting started

**1. Activate the virtual environment**

```bash
source .venv/Scripts/activate   # Windows (Git Bash)
source .venv/bin/activate       # macOS / Linux
```

**2. Install the project and dev dependencies**

```bash
pip install -e ".[dev]"
```

**3. Start Qdrant**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. Run the app**

```bash
python -m nndraw
```

## Development

```bash
# Run all tests
python -m pytest

# Run tests with a coverage report
python -m pytest --cov=nndraw
```

> [!NOTE]
> This project follows a test-driven development workflow — tests are written before implementation. The test layout mirrors `src/nndraw/` exactly, with one test file per module.

## Project structure

```
src/nndraw/
  linalg/   — custom linear algebra (Vector, Matrix, ...)
  nn/       — neural network built on linalg/
  db/       — Qdrant repository layer
  ui/       — Pygame interactive canvas
tests/      — mirrors src/nndraw/, one file per module
```

## Key conventions

- **No third-party math** — `linalg/` is entirely hand-rolled. Never import numpy or scipy for math operations.
- **One class per file** — file names match their class in `snake_case`.
- **TDD** — always write the test first, then implement.
