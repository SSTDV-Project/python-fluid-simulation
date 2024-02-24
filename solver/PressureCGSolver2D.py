import cupy as cp
from numba import cuda

from .SolidFraction2D import compute_solid_frac, edge_in_fraction

@cuda.jit
def initialize_solver_kernel(cell_size, gres, vx, vy, sphi, sv, lphi, b, wx, wy):
    """
    gres: grid resolution [W, H]
    vx: velocity x-axis [W+1, H]
    vy: velocity y-axis [W, H+1]
    sphi: solid levelset [2W+1, 2H+1]
    sv: solid velocity [2W+1, 2H+1, 2]
    lphi: fluid levelset [W, H]
    b, wx, wy: internal variables
    """
    x, y = cuda.grid(2)
    if x == 0 or x >= gres[0] - 1 or y == 0 or y >= gres[1]-1:
        # ignore boundary cells
        return

    if lphi[x,y] >= 0:
        # ignore non-fluid cells
        b[x,y] = 0
        return
    
    b_val = 0
    b_val += wx[x+1,y] * vx[x+1,y] / cell_size[0]
    if wx[x+1,y] < 1:
        b_val -= wx[x+1,y] * sv[2*x+2, 2*y+1, 0] / cell_size[0]
    
    b_val -= wx[x,y] * vx[x,y] / cell_size[0]
    if wx[x,y] < 1:
        b_val += wx[x,y] * sv[2*x, 2*y+1, 0] / cell_size[0]
    
    b_val += wy[x,y+1] * vy[x,y+1] / cell_size[1]
    if wy[x,y+1] < 1:
        b_val -= wy[x,y+1] * sv[2*x+1, 2*y+2, 1] / cell_size[1]
    
    b_val -= wy[x,y] * vy[x,y] / cell_size[1]
    if wy[x,y] < 1:
        b_val += wy[x,y] * sv[2*x+1, 2*y, 1] / cell_size[1]
    
    b[x,y] = b_val

@cuda.jit
def matvecmul_kernel(gres, v, out, wx, wy, lphi):
    x, y = cuda.grid(2)
    if x == 0 or x >= gres[0] - 1 or y == 0 or y >= gres[1]-1:
        # ignore boundary cells
        return
    
    phi = lphi[x,y]
    if phi >= 0:
        out[x,y] = 0
        # ignore non-fluid cells
        return
    
    val = 0.0
    diag = 0.0
    
    nphi = lphi[x+1,y]
    w = wx[x+1,y]
    if nphi < 0:
        val -= w * v[x+1,y]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    nphi = lphi[x-1,y]
    w = wx[x,y]
    if nphi < 0:
        val -= w * v[x-1,y]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    nphi = lphi[x,y+1]
    w = wy[x,y+1]
    if nphi < 0:
        val -= w * v[x,y+1]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    nphi = lphi[x,y-1]
    w = wy[x,y]
    if nphi < 0:
        val -= w * v[x,y-1]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
    
    val += diag * v[x,y]
    
    out[x,y] = val

@cuda.jit
def apply_pressure_kernel(gres, cell_size, vx, vy, pv, wx, wy, sv, lphi):
    x, y = cuda.grid(2)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1:
        # ignore boundary cells
        return
    
    if (lphi[x,y] < 0 or lphi[x-1,y] < 0):
        phix = min(1, max(0.01, edge_in_fraction(lphi[x,y], lphi[x-1,y])))
        new_vx = vx[x,y]
        new_vx += (pv[x,y] - pv[x-1,y]) * cell_size[0] / phix
        new_vx = wx[x,y]*new_vx + (1-wx[x,y])*sv[2*x,2*y+1,0]
        vx[x,y] = new_vx
    if (lphi[x,y] < 0 or lphi[x,y-1] < 0):
        phiy = min(1, max(0.01, edge_in_fraction(lphi[x,y], lphi[x,y-1])))
        new_vy = vy[x,y]
        new_vy += (pv[x,y] - pv[x,y-1]) * cell_size[1] / phiy
        new_vy = wy[x,y]*new_vy + (1-wy[x,y])*sv[2*x+1,2*y,1]
        vy[x,y] = new_vy

def initialize_solver(cell_size, gres, vx, vy, sphi, sv, lphi, b, wx, wy):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    initialize_solver_kernel[blocks, THREAD2](cell_size, gres, vx, vy, sphi, sv, lphi, b, wx, wy)

def matvecmul(gres, v, out, wx, wy, lphi):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    matvecmul_kernel[blocks, THREAD2](gres, v, out, wx, wy, lphi)

def apply_pressure(gres, cell_size, vx, vy, pv, wx, wy, sv, lphi):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    apply_pressure_kernel[blocks, THREAD2](gres, cell_size, vx, vy, pv, wx, wy, sv, lphi)

class PressureCGSolver2D:
    def __init__(self, buf, gres, bound_size):
        self.gres = gres
        self.cell_size = bound_size / gres
        self.buf = buf
        self.x = cp.zeros(gres.get(), dtype=cp.float64)
        self.wx = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wy = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, vx, vy, sphi, sv, lphi, wx=None, wy=None, tol=1e-3):
        if wx is None or wy is None:
            compute_solid_frac(self.gres, sphi, self.wx, self.wy)
            wx = self.wx
            wy = self.wy
        self.x *= 0.0
        initialize_solver(self.cell_size, self.gres, vx, vy, sphi, sv, lphi, self.buf.b, wx, wy)
        matvecmul(self.gres, self.x, self.buf.q, wx, wy, lphi)
        self.buf.d[:] = self.buf.b - self.buf.q
        self.buf.r[:] = self.buf.d
        self.delta = cp.sum(self.buf.r ** 2).item()
        if not self.delta < tol ** 2:
            for i in range(self.max_iter):
                matvecmul(self.gres, self.buf.d, self.buf.q, wx, wy, lphi)
                self.alpha = self.delta / cp.sum(self.buf.d * self.buf.q).item()
                self.x += self.alpha * self.buf.d
                self.buf.r -= self.alpha * self.buf.q

                old_delta = self.delta
                self.delta = cp.sum(self.buf.r ** 2).item()
                if self.delta < tol ** 2:
                    break
                self.beta = self.delta / old_delta
                self.buf.d[:] = self.buf.r + self.beta * self.buf.d
        # self.x : -pressure * dt / rho / cell_vol
        apply_pressure(self.gres, self.cell_size, vx, vy, self.x, wx, wy, sv, lphi)
