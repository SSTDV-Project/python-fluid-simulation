import cupy as cp
from numba import cuda

from .SolidFraction3D import compute_solid_frac, edge_in_fraction

@cuda.jit
def initialize_solver_kernel(cell_size, gres, vx, vy, vz, sphi, sv, lphi, b, wx, wy, wz):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= gres[0] - 1 or y == 0 or y >= gres[1]-1 or z == 0 or z >= gres[2]-1:
        # ignore boundary cells
        return
    
    if lphi[x,y,z] >= 0:
        # ignore non-fluid cells
        b[x,y,z] = 0
        return
    
    b_val = 0
    
    # +x
    b_val += wx[x+1,y,z] * vx[x+1,y,z] / cell_size[0]
    if wx[x+1,y,z] < 1:
        b_val -= wx[x+1,y,z] * sv[2*x+2, 2*y+1, 2*z+1, 0] / cell_size[0]
    
    # -x
    b_val -= wx[x,y,z] * vx[x,y,z] / cell_size[0]
    if wx[x,y,z] < 1:
        b_val += wx[x,y,z] * sv[2*x, 2*y+1, 2*z+1, 0] / cell_size[0]
    
    # +y
    b_val += wy[x,y+1,z] * vy[x,y+1,z] / cell_size[1]
    if wy[x,y+1,z] < 1:
        b_val -= wy[x,y+1,z] * sv[2*x+1, 2*y+2, 2*z+1, 1] / cell_size[1]
    
    # -y
    b_val -= wy[x,y,z] * vy[x,y,z] / cell_size[1]
    if wy[x,y,z] < 1:
        b_val += wy[x,y,z] * sv[2*x+1, 2*y, 2*z+1, 1] / cell_size[1]
    
    # +z
    b_val += wz[x,y,z+1] * vz[x,y,z+1] / cell_size[2]
    if wz[x,y,z+1] < 1:
        b_val -= wz[x,y,z+1] * sv[2*x+1, 2*y+1, 2*z+2, 2] / cell_size[2]
    
    # -z
    b_val -= wz[x,y,z] * vz[x,y,z] / cell_size[2]
    if wz[x,y,z] < 1:
        b_val += wz[x,y,z] * sv[2*x+1, 2*y+1, 2*z, 2] / cell_size[2]
    
    b[x,y,z] = b_val

@cuda.jit
def matvecmul_kernel(gres, v, out, wx, wy, wz, lphi):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= gres[0] - 1 or y == 0 or y >= gres[1]-1 or z == 0 or z >= gres[2]-1:
        # ignore boundary cells
        return
    
    phi = lphi[x,y,z]
    if phi >= 0:
        out[x,y,z] = 0
        # ignore non-fluid cells
        return
    
    val = 0.0
    diag = 0.0
    
    # +x
    nphi = lphi[x+1,y,z]
    w = wx[x+1,y,z]
    if nphi < 0:
        val -= w * v[x+1,y,z]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac

    # -x
    nphi = lphi[x-1,y,z]
    w = wx[x,y,z]
    if nphi < 0:
        val -= w * v[x-1,y,z]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    # +y
    nphi = lphi[x,y+1,z]
    w = wy[x,y+1,z]
    if nphi < 0:
        val -= w * v[x,y+1,z]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    # -y
    nphi = lphi[x,y-1,z]
    w = wy[x,y,z]
    if nphi < 0:
        val -= w * v[x,y-1,z]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    # +z
    nphi = lphi[x,y,z+1]
    w = wz[x,y,z+1]
    if nphi < 0:
        val -= w * v[x,y,z+1]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
        
    # -z
    nphi = lphi[x,y,z-1]
    w = wz[x,y,z]
    if nphi < 0:
        val -= w * v[x,y,z-1]
        diag += w
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += w / frac
    
    val += diag * v[x,y,z]
    
    out[x,y,z] = val

@cuda.jit
def apply_pressure_kernel(gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi):
    x, y, z = cuda.grid(3)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1 or z == 0 or z > gres[2]-1:
        # ignore boundary cells
        return
    
    if (lphi[x,y,z] < 0 or lphi[x-1,y,z] < 0):
        phix = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x-1,y,z])))
        new_vx = vx[x,y,z] + (pv[x,y,z] - pv[x-1,y,z]) * cell_size[0] / phix
        new_vx = wx[x,y,z]*new_vx + (1-wx[x,y,z])*sv[2*x,2*y+1,2*z+1,0]
        vx[x,y,z] = new_vx
    if (lphi[x,y,z] < 0 or lphi[x,y-1,z] < 0):
        phiy = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y-1,z])))
        new_vy = vy[x,y,z] + (pv[x,y,z] - pv[x,y-1,z]) * cell_size[1] / phiy
        new_vy = wy[x,y,z]*new_vy + (1-wy[x,y,z])*sv[2*x+1,2*y,2*z+1,1]
        vy[x,y,z] = new_vy
    if (lphi[x,y,z] < 0 or lphi[x,y,z-1] < 0):
        phiz = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y,z-1])))
        new_vz = vz[x,y,z] + (pv[x,y,z] - pv[x,y,z-1]) * cell_size[2] / phiz
        new_vz = wz[x,y,z]*new_vz + (1-wz[x,y,z])*sv[2*x+1,2*y+1,2*z,2]
        vz[x,y,z] = new_vz

def initialize_solver(cell_size, gres, vx, vy, vz, sphi, sv, lphi, b, wx, wy, wz):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    initialize_solver_kernel[blocks, THREAD3](cell_size, gres, vx, vy, vz, sphi, sv, lphi, b, wx, wy, wz)

def matvecmul(gres, v, out, wx, wy, wz, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    matvecmul_kernel[blocks, THREAD3](gres, v, out, wx, wy, wz, lphi)

def apply_pressure(gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    apply_pressure_kernel[blocks, THREAD3](gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi)
    
class PressureCGSolver3D:
    def __init__(self, buf, gres, bound_size):
        self.gres = gres
        self.cell_size = bound_size / gres
        self.buf = buf
        # self.d = cp.zeros(gres.get(), dtype=cp.float64)
        # self.r = cp.zeros(gres.get(), dtype=cp.float64)
        # self.q = cp.zeros(gres.get(), dtype=cp.float64)
        self.x = cp.zeros(gres.get(), dtype=cp.float64)
        # self.b = cp.zeros(gres.get(), dtype=cp.float64)
        self.wx = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wy = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wz = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, vx, vy, vz, sphi, sv, lphi, wx=None, wy=None, wz=None, tol=1e-3):
        if wx is None or wy is None or wz is None:
            compute_solid_frac(self.gres, sphi, self.wx, self.wy, self.wz)
            wx = self.wx
            wy = self.wy
            wz = self.wz
        self.x *= 0.0
        initialize_solver(self.cell_size, self.gres, vx, vy, vz, sphi, sv, lphi, self.buf.b, wx, wy, wz)
        
        matvecmul(self.gres, self.x, self.buf.q, wx, wy, wz, lphi)
        self.buf.d[:] = self.buf.b - self.buf.q
        self.buf.r[:] = self.buf.d
        self.delta = cp.sum(self.buf.r ** 2).item()
        # print(self.delta)
        if not self.delta < tol ** 2:
            for i in range(self.max_iter):
                matvecmul(self.gres, self.buf.d, self.buf.q, wx, wy, wz, lphi)
                cuda.synchronize()
                
                self.alpha = self.delta / cp.sum(self.buf.d * self.buf.q).item()
                self.x += self.alpha * self.buf.d
                self.buf.r -= self.alpha * self.buf.q

                old_delta = self.delta
                self.delta = cp.sum(self.buf.r ** 2).item()
                # print(self.delta)
                if self.delta < tol ** 2:
                    break
                self.beta = self.delta / old_delta
                self.buf.d[:] = self.buf.r + self.beta * self.buf.d
            else:
                raise ValueError("Failed to converge!")
        
        # self.x : -pressure * dt / rho / cell_vol
        apply_pressure(self.gres, self.cell_size, vx, vy, vz, self.x, wx, wy, wz, sv, lphi)
