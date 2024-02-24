import cupy as cp

class CGSolverBuffer:
    def __init__(self, gres):
        self.d = cp.zeros(gres.get(), dtype=cp.float64)
        self.r = cp.zeros(gres.get(), dtype=cp.float64)
        self.q = cp.zeros(gres.get(), dtype=cp.float64)
        self.b = cp.zeros(gres.get(), dtype=cp.float64)
