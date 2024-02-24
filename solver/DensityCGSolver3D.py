import math

import cupy as cp
from numba import cuda

from .SolidFraction3D import compute_solid_frac, edge_in_fraction

@cuda.jit
def initialize_density_kernel(bound_min, cell_size, gres, px, pm, pvol, gm, gvol):
    P = cuda.grid(1)
    if P >= px.shape[0]:
        return
    
    m = pm[P]
    x = cuda.local.array(3, dtype=cp.float64)
    gi = cuda.local.array(3, dtype=cp.int64)    # grid index
    gx = cuda.local.array(3, dtype=cp.float64)  # grid position
    w = cuda.local.array(3, dtype=cp.float64)   # weight |gx - x| / cell_size
    for d in range(3):
        x[d] = px[P,d]
        gi[d] = math.floor((x[d] - bound_min[d]) / cell_size[d] - 0.5)
        gx[d] = (gi[d] + 0.5) * cell_size[d] + bound_min[d]
        w[d] = abs(gx[d] - x[d]) / cell_size[d]
    
    for ix in [0,1]:
        for iy in [0,1]:
            for iz in [0,1]:
                gix = max(0, min(gres[0]-1, gi[0] + ix))
                giy = max(0, min(gres[1]-1, gi[1] + iy))
                giz = max(0, min(gres[2]-1, gi[2] + iz))
                wx = ix + ((-1)**ix) * (1 - w[0])
                wy = iy + ((-1)**iy) * (1 - w[1])
                wz = iz + ((-1)**iz) * (1 - w[2])
                weight = wx * wy * wz
                cuda.atomic.add(gm, (gix, giy, giz), weight * m)
                cuda.atomic.add(gvol, (gix, giy, giz), weight * pvol)
            
@cuda.jit
def fix_volume_kernel(cvol, dx, gres, lvol, gvol, sphi, lphi, wx, wy, wz):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= gres[0]-1 or y == 0 or y >= gres[1]-1 or z == 0 or z >= gres[2]-1:
        return

    # fluid_vol = (
    #     lvol[2*x+1, 2*y+1, 2*z+1] 
    #     + (1.0/2.0)*(lvol[2*x+2,2*y+1,2*z+1] + lvol[2*x,2*y+1,2*z+1]
    #                  + lvol[2*x+1,2*y+2,2*z+1] + lvol[2*x+1,2*y,2*z+1]
    #                  + lvol[2*x+1,2*y+1,2*z+2] + lvol[2*x+1,2*y+1,2*z])
    #     + (1.0/4.0)*(lvol[2*x+2,2*y+2,2*z+1] + lvol[2*x,2*y+2,2*z+1]
    #                  + lvol[2*x+2,2*y,2*z+1] + lvol[2*x,2*y,2*z+1]
    #                  + lvol[2*x+2,2*y+1,2*z+2] + lvol[2*x,2*y+1,2*z+2]
    #                  + lvol[2*x+2,2*y+1,2*z] + lvol[2*x,2*y+1,2*z]
    #                  + lvol[2*x+1,2*y+2,2*z+2] + lvol[2*x+1,2*y+2,2*z]
    #                  + lvol[2*x+1,2*y,2*z+2] + lvol[2*x+1,2*y,2*z])
    #     + (1.0/8.0)*(lvol[2*x+2,2*y+2,2*z+2] + lvol[2*x+2,2*y,2*z+2]
    #                  + lvol[2*x,2*y+2,2*z+2] + lvol[2*x,2*y,2*z+2]
    #                  + lvol[2*x+2,2*y+2,2*z] + lvol[2*x+2,2*y,2*z]
    #                  + lvol[2*x,2*y+2,2*z] + lvol[2*x,2*y,2*z])
    # )
    fluid_vol = gvol[x,y,z]

    # threshold = cvol * 0.2
    near_solid = sphi[2*x+1, 2*y+1, 2*z+1] < dx
    fluid_internal = (
        (lphi[x,y,z] < 0)
        and (lphi[x+1,y,z] < 0)
        and (lphi[x-1,y,z] < 0)
        and (lphi[x,y+1,z] < 0)
        and (lphi[x,y-1,z] < 0)
        and (lphi[x,y,z+1] < 0)
        and (lphi[x,y,z-1] < 0)
    )
    if fluid_internal and not near_solid:
        fluid_vol = cvol
    
    nonsolid_volfrac = (
        wx[x,y,z]
        + wx[x+1,y,z]
        + wy[x,y,z]
        + wy[x,y+1,z]
        + wz[x,y,z]
        + wz[x,y,z+1]
    ) / 6
    gvol[x,y,z] = min(fluid_vol, cvol * nonsolid_volfrac)

@cuda.jit
def initialize_solver_kernel(rho0, cvol, dt, gres, gm, gvol, lphi, wx, wy, wz, b):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= gres[0]-1 or y == 0 or y >= gres[1]-1 or z == 0 or z >= gres[2]-1:
        return
    
    if lphi[x,y,z] >= 0:
        # ignore non-fluid cells
        b[x,y,z] = 0
        return
    
    nonsolid_frac = (
        wx[x,y,z]
        + wx[x+1,y,z]
        + wy[x,y,z]
        + wy[x,y+1,z]
        + wz[x,y,z]
        + wz[x,y,z+1]
    ) / 6
    solid_vol = (1 - nonsolid_frac) * cvol
    solid_mass = rho0 * solid_vol
    
    cell_mass = gm[x,y,z] + solid_mass
    cell_vol = gvol[x,y,z] + solid_vol
    density_frac = cell_mass / max(cell_vol, 1e-10) / rho0
    if cell_mass < 1e-10:
        density_frac = 1

    density_frac = max(0.5, min(1.5, density_frac))
    b[x,y,z] = (1 - density_frac) / dt

@cuda.jit
def matvecmul_kernel(gres, v, out, wx, wy, wz, lphi):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= gres[0]-1 or y == 0 or y >= gres[1]-1 or z == 0 or z >= gres[2]-1:
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
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    # -x
    nphi = lphi[x-1,y,z]
    w = wx[x,y,z]
    if nphi < 0:
        val -= w * v[x-1,y,z]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    # +y
    nphi = lphi[x,y+1,z]
    w = wy[x,y+1,z]
    if nphi < 0:
        val -= w * v[x,y+1,z]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    # -y
    nphi = lphi[x,y-1,z]
    w = wy[x,y,z]
    if nphi < 0:
        val -= w * v[x,y-1,z]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    # +z
    nphi = lphi[x,y,z+1]
    w = wz[x,y,z+1]
    if nphi < 0:
        val -= w * v[x,y,z+1]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    # -z
    nphi = lphi[x,y,z-1]
    w = wz[x,y,z+1]
    if nphi < 0:
        val -= w * v[x,y,z-1]
        diag += 1
    else:
        frac = min(1, max(0.01, phi / (phi - nphi)))
        diag += 1 / frac
    
    val += diag * v[x,y,z]
    
    out[x,y,z] = val

@cuda.jit
def compute_displacement_kernel(gres, dt, cell_size, dx, dy, dz, pv, lphi):
    x, y, z = cuda.grid(3)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1 or z == 0 or z > gres[2]-1:
        # ignore boundary cells
        return
    
    phix = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x-1,y,z])))
    phiy = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y-1,z])))
    phiz = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y,z-1])))

    dx[x,y,z] = (pv[x,y,z] - pv[x-1,y,z]) * dt * cell_size[0] / phix
    dy[x,y,z] = (pv[x,y,z] - pv[x,y-1,z]) * dt * cell_size[1] / phiy
    dz[x,y,z] = (pv[x,y,z] - pv[x,y,z-1]) * dt * cell_size[2] / phiz

@cuda.jit
def apply_displacement_kernel(px, dx, bound_min, cell_size, grid_bias, axis):
    # See g2p
    P = cuda.grid(1)
    if P >= px.shape[0]:
        return
    
    x = cuda.local.array(3, dtype=cp.float64)
    gi = cuda.local.array(3, dtype=cp.int64)    # grid index
    gx = cuda.local.array(3, dtype=cp.float64)  # grid position
    w = cuda.local.array(3, dtype=cp.float64)   # weight |gx - x| / cell_size
    for d in range(3):
        x[d] = px[P,d]
        gi[d] = math.floor((x[d] - bound_min[d]) / cell_size[d] - grid_bias[d])
        gx[d] = (gi[d] + grid_bias[d]) * cell_size[d] + bound_min[d]
        w[d] = abs(gx[d] - x[d]) / cell_size[d]
    
    for ix in [0,1]:
        for iy in [0,1]:
            for iz in [0,1]:
                gix = max(0, min(dx.shape[0]-1, gi[0] + ix))
                giy = max(0, min(dx.shape[1]-1, gi[1] + iy))
                giz = max(0, min(dx.shape[2]-1, gi[2] + iz))
                wx = ix + ((-1)**ix) * (1 - w[0])
                wy = iy + ((-1)**iy) * (1 - w[1])
                wz = iz + ((-1)**iz) * (1 - w[2])
                weight = wx * wy * wz
                px[P,axis] += weight * dx[gix, giy, giz]

def initialize_density(bound_min, cell_size, gres, px, pm, pvol, gm, gvol, sphi, lphi):
    P = px.shape[0]
    THREAD1 = 512
    blocks = (P - 1) // THREAD1 + 1
    
    initialize_density_kernel[blocks, THREAD1](bound_min, cell_size, gres, px, pm, pvol, gm, gvol)
    
def fix_volume(cell_size, gres, lvol, gvol, sphi, lphi, wx, wy, wz):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    
    cvol = cp.prod(cell_size).item()
    dx = cp.min(cell_size).item()
    fix_volume_kernel[blocks, THREAD3](cvol, dx, gres, lvol, gvol, sphi, lphi, wx, wy, wz)
    
def initialize_solver(rho0, dt, gres, cell_size, gm, gvol, lphi, wx, wy, wz, b):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    
    cvol = cp.prod(cell_size).item()
    initialize_solver_kernel[blocks, THREAD3](rho0, cvol, dt, gres, gm, gvol, lphi, wx, wy, wz, b)

def matvecmul(gres, v, out, wx, wy, wz, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    matvecmul_kernel[blocks, THREAD3](gres, v, out, wx, wy, wz, lphi)

def compute_displacement(gres, dt, cell_size, dx, dy, dz, pv, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    compute_displacement_kernel[blocks, THREAD3](gres, dt, cell_size, dx, dy, dz, pv, lphi)

def apply_displacement(px, dx, bound_min, cell_size, grid_bias, axis):
    P = px.shape[0]
    THREAD1 = 512
    blocks = (P - 1) // THREAD1 + 1
    
    apply_displacement_kernel[blocks, THREAD1](px, dx, bound_min, cell_size, grid_bias, axis)
    
class DensityCGSolver3D:
    def __init__(self, buf, gres, bound_min, bound_size):
        self.gres = gres
        self.bound_min = bound_min
        self.cell_size = bound_size / gres
        self.bias_x = cp.array([0,0.5,0.5], dtype=cp.float64)
        self.bias_y = cp.array([0.5,0,0.5], dtype=cp.float64)
        self.bias_z = cp.array([0.5,0.5,0], dtype=cp.float64)
        
        self.buf = buf
        self.m = cp.zeros(gres.get(), dtype=cp.float64)
        self.vol = cp.zeros(gres.get(), dtype=cp.float64)
        # self.d = cp.zeros(gres.get(), dtype=cp.float64)
        # self.r = cp.zeros(gres.get(), dtype=cp.float64)
        # self.q = cp.zeros(gres.get(), dtype=cp.float64)
        self.x = cp.zeros(gres.get(), dtype=cp.float64)
        # self.b = cp.zeros(gres.get(), dtype=cp.float64)
        self.wx = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wy = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.wz = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.dx = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.dy = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.dz = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, rho0, dt, px, pm, pvol, vx, vy, vz, sphi, sv, lphi, lvol, wx=None, wy=None, wz=None, tol=1e-3):
        if wx is None or wy is None or wz is None:
            compute_solid_frac(self.gres, sphi, self.wx, self.wy, self.wz)
            wx = self.wx
            wy = self.wy
            wz = self.wz
        self.m *= 0
        self.vol *= 0
        self.x *= 0
        initialize_density(self.bound_min, self.cell_size, self.gres, px, pm, pvol, self.m, self.vol, sphi, lphi)
        fix_volume(self.cell_size, self.gres, lvol, self.vol, sphi, lphi, wx, wy, wz)
        initialize_solver(rho0, dt, self.gres, self.cell_size, self.m, self.vol, lphi, wx, wy, wz, self.buf.b)
        
        matvecmul(self.gres, self.x, self.buf.q, wx, wy, wz, lphi)
        self.buf.d[:] = self.buf.b - self.buf.q
        self.buf.r[:] = self.buf.d
        self.delta = cp.sum(self.buf.r ** 2).item()
        if not self.delta < tol ** 2:
            for i in range(self.max_iter):
                matvecmul(self.gres, self.buf.d, self.buf.q, wx, wy, wz, lphi)
                cuda.synchronize()
                
                self.alpha = self.delta / cp.sum(self.buf.d * self.buf.q).item()
                self.x += self.alpha * self.buf.d
                self.buf.r -= self.alpha * self.buf.q

                old_delta = self.delta
                self.delta = cp.sum(self.buf.r ** 2).item()
                if self.delta < tol ** 2:
                    break
                self.beta = self.delta / old_delta
                self.buf.d[:] = self.buf.r + self.beta * self.buf.d
            else:
                raise ValueError("Failed to converge!")
        # self.x : -pressure * dt / rho / dx^2
        compute_displacement(self.gres, dt, self.cell_size, self.dx, self.dy, self.dz, self.x, lphi)
        apply_displacement(px, self.dx, self.bound_min, self.cell_size, self.bias_x, 0)
        apply_displacement(px, self.dy, self.bound_min, self.cell_size, self.bias_y, 1)
        apply_displacement(px, self.dz, self.bound_min, self.cell_size, self.bias_z, 2)
