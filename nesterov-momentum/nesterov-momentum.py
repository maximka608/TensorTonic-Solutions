import numpy as np


def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    new_v = momentum * np.array(v) + lr * np.array(grad)
    new_w = np.array(w) - new_v
    return new_w, new_v