import numpy as np

class Kernels:

    def triang(xi, w):
        R = np.maximum(0.0, 1.0 - np.abs(xi[:, None] - xi[None, :]) / w)
        return R

    def sqexp(xi, w):
        R = np.exp(-0.5 * ((xi[:, None] - xi[None, :]) / w)**2)
        jitter = 1e-15
        R[np.diag_indices_from(R)] += jitter
        return R