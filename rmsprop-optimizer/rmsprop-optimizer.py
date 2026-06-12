import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    new_s = beta * np.array(s) + (1-beta) * np.pow(g, 2)
    new_w = np.array(w) - (lr / np.sqrt(new_s + eps)) * np.array(g)
    return new_w, new_s