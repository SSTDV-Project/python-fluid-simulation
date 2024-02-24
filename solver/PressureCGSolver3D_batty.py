import cupy as cp
from numba import cuda
import cupyx
from .SolidFraction3D import compute_solid_frac, edge_in_fraction
import cupyx.scipy.sparse.linalg
import cupyx.scipy.sparse
def fraction_inside(cphi,rphi):
    return cphi / (cphi - rphi)
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
    b_val -= wx[x+1,y,z] * vx[x+1,y,z] / cell_size[0]
    if wx[x+1,y,z] < 1:
        b_val += wx[x+1,y,z] * sv[2*x+2, 2*y+1, 2*z+1, 0] / cell_size[0]
    
    # -x
    b_val += wx[x,y,z] * vx[x,y,z] / cell_size[0]
    if wx[x,y,z] < 1:
        b_val -= wx[x,y,z] * sv[2*x, 2*y+1, 2*z+1, 0] / cell_size[0]
    
    # +y
    b_val -= wy[x,y+1,z] * vy[x,y+1,z] / cell_size[1]
    if wy[x,y+1,z] < 1:
        b_val += wy[x,y+1,z] * sv[2*x+1, 2*y+2, 2*z+1, 1] / cell_size[1]
    
    # -y
    b_val += wy[x,y,z] * vy[x,y,z] / cell_size[1]
    if wy[x,y,z] < 1:
        b_val -= wy[x,y,z] * sv[2*x+1, 2*y, 2*z+1, 1] / cell_size[1]
    
    # +z
    b_val -= wz[x,y,z+1] * vz[x,y,z+1] / cell_size[2]
    if wz[x,y,z+1] < 1:
        b_val += wz[x,y,z+1] * sv[2*x+1, 2*y+1, 2*z+2, 2] / cell_size[2]
    
    # -z
    b_val += wz[x,y,z] * vz[x,y,z] / cell_size[2]
    if wz[x,y,z] < 1:
        b_val -= wz[x,y,z] * sv[2*x+1, 2*y+1, 2*z, 2] / cell_size[2]
    
    b[x,y,z] = b_val

@cuda.jit
def matvecmul_kernel(cell_size, gres, v, out, wx, wy, wz, lphi, dt):
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
    term = wx[x+1,y,z]*dt/(cell_size[0])**(0.5)
    r_phi = lphi[x+1,y,z]
    if r_phi < 0:
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
def apply_pressure_kernel(dt, gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi):
    x, y, z = cuda.grid(3)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1 or z == 0 or z > gres[2]-1:
        # ignore boundary cells
        return
    
    if (lphi[x,y,z] < 0 or lphi[x-1,y,z] < 0):
        phix = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x-1,y,z])))
        new_vx = vx[x,y,z] - (pv[x,y,z] - pv[x-1,y,z]) * dt / cell_size[0] / phix
        new_vx = wx[x,y,z]*new_vx + (1-wx[x,y,z])*sv[2*x,2*y+1,2*z+1,0]
        vx[x,y,z] = new_vx
    if (lphi[x,y,z] < 0 or lphi[x,y-1,z] < 0):
        phiy = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y-1,z])))
        new_vy = vy[x,y,z] - (pv[x,y,z] - pv[x,y-1,z]) * dt / cell_size[1] / phiy
        new_vy = wy[x,y,z]*new_vy + (1-wy[x,y,z])*sv[2*x+1,2*y,2*z+1,1]
        vy[x,y,z] = new_vy
    if (lphi[x,y,z] < 0 or lphi[x,y,z-1] < 0):
        phiz = min(1, max(0.01, edge_in_fraction(lphi[x,y,z], lphi[x,y,z-1])))
        new_vz = vz[x,y,z] - (pv[x,y,z] - pv[x,y,z-1]) * dt /  cell_size[2] / phiz
        new_vz = wz[x,y,z]*new_vz + (1-wz[x,y,z])*sv[2*x+1,2*y+1,2*z,2]
        vz[x,y,z] = new_vz

def initialize_solver(cell_size, gres, vx, vy, vz, sphi, sv, lphi, b, wx, wy, wz):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    initialize_solver_kernel[blocks, THREAD3](cell_size, gres, vx, vy, vz, sphi, sv, lphi, b, wx, wy, wz)

def matvecmul(cell_size, gres, v, out, wx, wy, wz, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    matvecmul_kernel[blocks, THREAD3](cell_size, gres, v, out, wx, wy, wz, lphi)

def apply_pressure(dt, gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    apply_pressure_kernel[blocks, THREAD3](dt, gres, cell_size, vx, vy, vz, pv, wx, wy, wz, sv, lphi)
    
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

    def solve(self, dt, vx, vy, vz, sphi, sv, lphi, wx=None, wy=None, wz=None, tol=1e-3):
        if wx is None or wy is None or wz is None:
            compute_solid_frac(self.gres, sphi, self.wx, self.wy, self.wz)
            wx = self.wx
            wy = self.wy
            wz = self.wz
        ni=int(self.gres[0])
        nj=int(self.gres[1])
        nk=int(self.gres[2])
        print(ni,nj,nk)
        self.x*=0
        size = int(ni*nj*nk)
        rhs= cp.zeros((size,1))
        matrix= cupyx.scipy.sparse.csr_matrix((size,size))
        u_weights, v_weights, w_weights = wx, wy, wz
        dx, dy, dz = self.cell_size[0],self.cell_size[1],self.cell_size[2]
        for k in range(1, nk-1):
            for j in range(1, nj-1):
                for i in range(1, ni-1):
                    index = i + ni*j + ni*nj*k
                    rhs[index] = 0;
                    #pressure[index] = 0;
                    centre_phi = lphi[i,j,k]
                    if centre_phi < 0:
                        #right neighbour
                        term = u_weights[i+1,j,k] * dt / (dx**0.5);
                        right_phi = lphi[i+1,j,k]
                        if right_phi < 0:
                            matrix[index, index]= term
                            matrix[index, index + 1] = -term;
                        else:
                            theta = fraction_inside(centre_phi, right_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta
                        rhs[index] -= u_weights[i+1,j,k] * vx[i+1,j,k] / dx;

                       #left neighbour
                        term = u_weights[i,j,k] * dt / (dx**0.5)
                        left_phi = lphi[i-1,j,k]
                        if left_phi < 0: 
                            matrix[index, index] = term;
                            matrix[index, index - 1]= -term
                       
                        else: 
                            theta = fraction_inside(centre_phi, left_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta
                       
                        rhs[index] += u_weights[i,j,k] * vx[i,j,k] / dx
                        
                        #top neighbour
                        term = v_weights[i,j+1,k] * dt / (dy**0.5);
                        top_phi = lphi[i,j+1,k]
                        if top_phi < 0:
                            matrix[index, index]= term
                            matrix[index, index + ni] = -term;
                        else:
                            theta = fraction_inside(centre_phi, top_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta
                        rhs[index] -= v_weights[i,j+1,k] * vy[i,j+1,k] / dy
                        
                        #bottom neighbour
                        term = v_weights[i,j,k] * dt / (dy**0.5)
                        bot_phi = lphi[i,j-1,k]
                        if bot_phi < 0: 
                            matrix[index, index] = term
                            matrix[index, index - ni] = -term

                        else: 
                            theta = fraction_inside(centre_phi, bot_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta

                        rhs[index] += v_weights[i,j,k]*vy[i,j,k] / dy
                        
                        #far neighbour
                        term = w_weights[i,j,k+1] * dt / (dz**0.5)
                        far_phi = lphi[i,j,k+1]
                        if far_phi < 0:
                            matrix[index, index]= term
                            matrix[index, index + ni*nj] = -term
                        else:
                            theta = fraction_inside(centre_phi, far_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta
                        rhs[index] -= w_weights[i,j,k+1] * vz[i,j,k+1] / dz

                        #near neighbour
                        term = w_weights[i,j,k] * dt / (dz**0.5)
                        near_phi = lphi[i-1,j,k]
                        if near_phi < 0: 
                            matrix[index, index] = term
                            matrix[index, index] = -term

                        else: 
                            theta = fraction_inside(centre_phi, near_phi)
                            if theta < 0.01: theta = 0.01
                            matrix[index, index] = term/theta
                        rhs[index] += w_weights[i,j,k] * vz[i,j,k] / dz
        x, iters = cupyx.scipy.sparse.linalg.cg(matrix, rhs, tol=1e-2)

        for k in range(1, nk-1):
            for j in range(1, nj-1):
                for i in range(1, ni-1):
                    index = i + j*ni + k*ni*nj
                    self.x[i,j,k] = x[index]
                    #print (ni,nj,nki,j,k)
        
        # self.x : -pressure * dt / rho / cell_vol
        apply_pressure(dt, self.gres, self.cell_size, vx, vy, vz, self.x, wx, wy, wz, sv, lphi)
