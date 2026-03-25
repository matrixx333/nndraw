# CLAUDE.md

## Project

`nndraw` — an interactive neural network decision boundary visualizer.
Built to learn Python, linear algebra, machine learning fundamentals, and Qdrant (vector DB) simultaneously.

## Setup

```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows (Git Bash)

# Install project + dev dependencies
pip install -e ".[dev]"

# Start Qdrant (requires Docker)
docker run -p 6333:6333 qdrant/qdrant
```

## Commands

```bash
# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=nndraw

# Run the application
python -m nndraw
```

## Structure

```
src/nndraw/
  linalg/   — custom linear algebra library (no third-party math libs)
  nn/       — neural network built on top of linalg/
  db/       — Qdrant repository layer
  ui/       — Pygame interactive canvas
tests/      — mirrors src/nndraw/ structure, one test file per module
```

## Conventions

- All linear algebra is hand-rolled — never import numpy/scipy for math
- Tests are written before implementation (TDD)
- One class per file, file name matches class name in snake_case
- User writes all Python code; Claude teaches and guides
