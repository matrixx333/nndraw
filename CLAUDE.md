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

## Skills

Below is a list of skills you have access to: 

- async-python-patterns
- claude-md-improver
- python-code-style
- python-design-patterns
- python-project-structure
- python-testing-patterns

## Conventions

- All linear algebra is hand-rolled — never import numpy/scipy for math
- Tests are written before implementation (TDD)
- One class per file, file name matches class name in snake_case
- User writes all Python code; Claude teaches and guides

## Teaching Style

At the start of every session:
1. Check `git status` and the test file for the current step to determine where we left off
2. Briefly recap what was just completed
3. State clearly what the next task is

When guiding through a task:
- Explain the concept first (what it is, why it matters, how it connects to what's already built)
- Then give a specific, concrete task for the user to attempt themselves
- Wait for the user to write the code before offering corrections or the answer
- Ask for the test before the implementation — always TDD

The current build order is in `docs/plan.md`. We are progressing through it step by step.
