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

def tanh(x: float) -> float:
    """
    Maps any value to (-1, 1). Zero-centered, unlike sigmoid — preferred for hidden layers
    because it helps gradients flow symmetrically during backpropagation.

    tanh(0) = 0, large positive → 1, large negative → -1.
    """
    e_x = math.exp(x)
    e_neg_x = math.exp(-x)
    return (e_x - e_neg_x) / (e_x + e_neg_x)

def tanh_derivative(x: float) -> float:
    """
    Slope of tanh at x. Used during backpropagation to compute weight adjustments.

    tanh_derivative(0) = 1. Approaches 0 for large |x| (vanishing gradient).
    """
    t = tanh(x)
    return 1 - (t ** 2)

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Like relu, but allows a small slope (alpha) for negative inputs instead of clamping to zero.
    Fixes the "dying ReLU" problem where neurons with always-negative inputs stop updating entirely.

    Returns x for positive inputs, alpha * x for negative inputs.
    """
    return x if x > 0 else alpha * x
    
def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
    """
    Slope of leaky_relu at x. Returns 1 for positive inputs (gradient passes through),
    alpha for negative inputs (small gradient still flows). Used during backpropagation.
    """
    return 1 if x > 0 else alpha