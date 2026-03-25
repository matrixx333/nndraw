import math

def sigmoid(x: float) -> float:
    """
    Maps any value to (0, 1). Used on the output layer for binary classification
    where the result is interpreted as a probability.

    sigmoid(0) = 0.5, large positive → 1, large negative → 0.
    """
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    """
    Slope of sigmoid at x. Used during backpropagation to compute how much
    to adjust weights in the output layer.

    sigmoid_derivative(0) = 0.25. Approaches 0 for large |x| (vanishing gradient).
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x: float) -> float:
    """
    Clamps negative values to zero, passes positive values through unchanged.
    Used on hidden layers — simple and avoids the vanishing gradient problem.
    """
    return x if x > 0 else 0.0

def relu_derivative(x: float) -> float:
    """
    Slope of relu at x. Returns 1 for positive inputs (gradient passes through),
    0 for negative inputs (gradient is blocked). Used during backpropagation
    to compute weight adjustments in hidden layers.
    """
    return 1 if x > 0 else 0.0