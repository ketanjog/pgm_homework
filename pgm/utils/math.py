"""
Useful functions
"""
import numpy as np


def sigmoid(n: np.ndarray):
    """
    Helper Function

    returns sigmoid(n)
    """
    n = n.astype(float)

    sig = 1.0 / (1.0 + np.exp(-n + 1e-7))

    sig[sig <= 0] = 1e-7
    sig[sig >= 1] = 1

    return sig
