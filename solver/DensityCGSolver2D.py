import math

import cupy as cp
from numba import cuda

from .SolidFraction2D import compute_solid_frac, edge_in_fraction

@cuda.jit
def initialize_density_kernel(bound_min, cell_size, gres, px, pm, pvol, gm, gvol):
    P = cuda.grid(1)
    if P >= px.shape[0]:
        return
    
    m = pm[P]
    x = cuda.local.array(2, dtype=cp.float64)
    gi = cuda.local.array(2, dtype=cp.int64)    # grid index
    gx = cuda.local.array(2, dtype=cp.float64)  # grid position
    w = cuda.local.array(2, dtype=cp.float64)   # weight |gx - x| / cell_size
    for d in range(2):
        x[d] = px[P,d]
        gi[d] = math.floor((x[d] - bound_min[d]) / cell_size[d] - 0.5)
        gx[d] = (gi[d] + 0.5) * cell_size[d] + bound_min[d]
        w[d] = abs(gx[d] - x[d]) / cell_size[d]
    
    for ix in [0,1]:
        for iy in [0,1]:
            gix = max(0, min(gres[0]-1, gi[0] + ix))
            giy = max(0, min(gres[1]-1, gi[1] + iy))
            wx = ix + ((-1)**ix) * (1 - w[0])
            wy = iy + ((-1)**iy) * (1 - w[1])
            weight = wx * wy
            cuda.atomic.add(gm, (gix, giy), weight * m)
            # cuda.atomic.add(gvol, (gix, giy), weight * pvol)
            
@cuda.jit
def fix_volume_kernel(cvol, dx, gres, lvol, gvol, sphi, lphi, wx, wy):
    x, y = cuda.grid(2)
    if x == 0 or x >= gres[0]-1 or y == 0 or y >= gres[1]-1:
        return

    fluid_vol = (
        lvol[2*x+1, 2*y+1] 
        + (1.0/2.0)*(lvol[2*x+2,2*y+1] + lvol[2*x,2*y+1] + lvol[2*x+1,2*y+2] + lvol[2*x+1,2*y])
        + (1.0/4.0)*(lvol[2*x+2,2*y+2] + lvol[2*x,2*y+2] + lvol[2*x+2,2*y] + lvol[2*x,2*y])
    )
    
    near_solid = sphi[2*x+1, 2*y+1] < dx
    fluid_internal = lphi[x,y] < 0 and (
        (lphi[x+1,y] < 0)
        and (lphi[x-1,y] < 0)
        and (lphi[x,y+1] < 0)
        and (lphi[x,y-1] < 0)
    )
    if fluid_internal and not near_solid:
        fluid_vol = cvol
    nonsolid_volfrac = (wx[x,y] + wx[x+1,y] + wy[x,y] + wy[x,y+1]) * 0.25
    gvol[x,y] = min(fluid_vol, cvol * nonsolid_volfrac)

@cuda.jit
def initialize_solver_kernel(rho0, cvol, dt, gres, gm, gvol, lphi, wx, wy, b):
    x, y = cuda.grid(2)
    if x == 0 or x >= gres[0] - 1 or y == 0 or y >= gres[1]-1:
        # ignore boundary cells
        return
    
    if lphi[x,y] >= 0:
        # ignore non-fluid cells
        b[x,y] = 0
        return
    
    
    nonsolid_frac = (wx[x,y] + wx[x+1,y] + wy[x,y] + wy[x,y+1]) * 0.25
    solid_vol = (1 - nonsolid_frac) * cvol
    solid_mass = rho0 * solid_vol
    
    cell_mass = gm[x,y] + solid_mass
    cell_vol = gvol[x,y] + solid_vol
    density_frac = cell_mass / max(cell_vol, 1e-10) / rho0
    if cell_mass < 1e-10:
        density_frac = 1
    
    density_frac = max(0.5, min(1.5, density_frac))
    b[x,y] = (1 - density_frac) / dt

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
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
        
    nphi = lphi[x-1,y]
    w = wx[x,y]
    if nphi < 0:
        val -= w * v[x-1,y]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
        
    nphi = lphi[x,y+1]
    w = wy[x,y+1]
    if nphi < 0:
        val -= w * v[x,y+1]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
        
    nphi = lphi[x,y-1]
    w = wy[x,y]
    if nphi < 0:
        val -= w * v[x,y-1]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    val += diag * v[x,y]
    
    out[x,y] = val

@cuda.jit
def compute_displacement_kernel(gres, dt, cell_size, dx, dy, pv, lphi):
    x, y = cuda.grid(2)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1:
        # ignore boundary cells
        return
    
    phix = min(1, max(0.01, edge_in_fraction(lphi[x,y], lphi[x-1,y])))
    phiy = min(1, max(0.01, edge_in_fraction(lphi[x,y], lphi[x,y-1])))

    dx[x,y] = (pv[x,y] - pv[x-1,y]) * dt * cell_size[0] / phix
    dy[x,y] = (pv[x,y] - pv[x,y-1]) * dt * cell_size[1] / phiy

# @cuda.jit
# def distribute_particles(rho_max, px, gm, gvol, bound_min, cell_size):
#     P = cuda.grid(1)
#     if P >= px.shape[0]:
#         return
    
#     x = cuda.local.array(2, dtype=cp.float64)
#     gi = cuda.local.array(2, dtype=cp.int64)    # grid index
#     for d in range(2):
#         x[d] = px[P,d]
#         gi[d] = math.floor((x[d] - bound_min[d]) / cell_size[d] - grid_bias[d])
    
#     grho = gm[gi[0], gi[1]] / gvol[gi[0], gi[1]]
#     if grho > rho_max:
#         for d in range(2):
#             x[d] += 

@cuda.jit
def apply_displacement_kernel(px, dx, bound_min, cell_size, grid_bias, axis):
    # See g2p
    P = cuda.grid(1)
    if P >= px.shape[0]:
        return
    
    x = cuda.local.array(2, dtype=cp.float64)
    gi = cuda.local.array(2, dtype=cp.int64)    # grid index
    gx = cuda.local.array(2, dtype=cp.float64)  # grid position
    w = cuda.local.array(2, dtype=cp.float64)   # weight |gx - x| / cell_size
    for d in range(2):
        x[d] = px[P,d]
        gi[d] = math.floor((x[d] - bound_min[d]) / cell_size[d] - grid_bias[d])
        gx[d] = (gi[d] + grid_bias[d]) * cell_size[d] + bound_min[d]
        w[d] = abs(gx[d] - x[d]) / cell_size[d]
    
    for ix in [0,1]:
        for iy in [0,1]:
            gix = max(0, min(dx.shape[0]-1, gi[0] + ix))
            giy = max(0, min(dx.shape[1]-1, gi[1] + iy))
            wx = ix + ((-1)**ix) * (1 - w[0])
            wy = iy + ((-1)**iy) * (1 - w[1])
            weight = wx * wy
            px[P,axis] += weight * dx[gix, giy]

def initialize_density(bound_min, cell_size, gres, px, pm, pvol, gm, gvol, sphi, lphi):
    P = px.shape[0]
    THREAD1 = 1024
    blocks = (P - 1) // THREAD1 + 1
    
    initialize_density_kernel[blocks, THREAD1](bound_min, cell_size, gres, px, pm, pvol, gm, gvol)
    
def fix_volume(cell_size, gres, lvol, gvol, sphi, lphi, wx, wy):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    
    cvol = cp.prod(cell_size).item()
    dx = cp.min(cell_size).item()
    fix_volume_kernel[blocks, THREAD2](cvol, dx, gres, lvol, gvol, sphi, lphi, wx, wy)
    
def initialize_solver(rho0, dt, gres, cell_size, gm, gvol, lphi, wx, wy, b):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)         
    
    cvol = cp.prod(cell_size).item()
    initialize_solver_kernel[blocks, THREAD2](rho0, cvol, dt, gres, gm, gvol, lphi, wx, wy, b)

def matvecmul(gres, v, out, wx, wy, lphi):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    matvecmul_kernel[blocks, THREAD2](gres, v, out, wx, wy, lphi)

def compute_displacement(gres, dt, cell_size, dx, dy, pv, lphi):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    compute_displacement_kernel[blocks, THREAD2](gres, dt, cell_size, dx, dy, pv, lphi)

def apply_displacement(px, dx, bound_min, cell_size, grid_bias, axis):
    P = px.shape[0]
    THREAD1 = 1024
    blocks = (P - 1) // THREAD1 + 1
    
    apply_displacement_kernel[blocks, THREAD1](px, dx, bound_min, cell_size, grid_bias, axis)
    
class DensityCGSolver2D:
    def __init__(self, buf, gres, bound_min, bound_size):
        self.gres = gres
        self.bound_min = bound_min
        self.cell_size = bound_size / gres
        self.bias_x = cp.array([0,0.5], dtype=cp.float64)
        self.bias_y = cp.array([0.5,0], dtype=cp.float64)
        
        self.m = cp.zeros(gres.get(), dtype=cp.float64)
        self.vol = cp.zeros(gres.get(), dtype=cp.float64)
        self.buf = buf
        self.x = cp.zeros(gres.get(), dtype=cp.float64)
        self.wx = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wy = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.dx = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.dy = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, rho0, dt, px, pm, pvol, vx, vy, sphi, sv, lphi, lvol, wx=None, wy=None, tol=1e-3):
        if wx is None or wy is None:
            compute_solid_frac(self.gres, sphi, self.wx, self.wy)
            wx = self.wx
            wy = self.wy
        self.m *= 0
        self.vol *= 0
        self.x *= 0
        initialize_density(self.bound_min, self.cell_size, self.gres, px, pm, pvol, self.m, self.vol, sphi, lphi)
        fix_volume(self.cell_size, self.gres, lvol, self.vol, sphi, lphi, wx, wy)
        initialize_solver(rho0, dt, self.gres, self.cell_size, self.m, self.vol, lphi, wx, wy, self.buf.b)
        
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
        # self.x : -pressure * dt / rho / dx^2
        compute_displacement(self.gres, dt, self.cell_size, self.dx, self.dy, self.x, lphi)
        apply_displacement(px, self.dx, self.bound_min, self.cell_size, self.bias_x, 0)
        apply_displacement(px, self.dy, self.bound_min, self.cell_size, self.bias_y, 1)
