import math

import cupy as cp
from numba import cuda

@cuda.jit
def initialize_solver_x_kernel(scale, mu, vx, vy, sphi, sv, vol, b):
    x, y = cuda.grid(2)
    if x == 0 or x >= b.shape[0] - 1 or y == 0 or y >= b.shape[1]-1:
        # ignore boundary cells
        return

    if sphi[2*x, 2*y+1] <= 0:
        # ignore solid cells
        b[x,y] = 0
        return
    
    vol_center = vol[2*x,   2*y+1]
    vol_right =  vol[2*x+1, 2*y+1]
    vol_left =   vol[2*x-1, 2*y+1]
    vol_top =    vol[2*x,   2*y+2]
    vol_bottom = vol[2*x,   2*y ]

    
    b_val = vx[x,y] * vol_center
    
    # vx +x
    if sphi[2*x+2, 2*y+1] <=0:
        b_val += 2 * scale * mu * vol_right * vx[x+1, y] # sv[2*x+2, 2*y+1, 0]
    # vx -x
    if sphi[2*x-2, 2*y+1] <= 0:
        b_val += 2 * scale * mu * vol_left * vx[x-1, y] # sv[2*x-2, 2*y+1, 0]
    # vx +y
    if sphi[2*x, 2*y+3] <= 0:
        b_val += scale * mu * vol_top * vx[x, y+1] # sv[2*x, 2*y+3, 0]
    # vx -y
    if sphi[2*x, 2*y-1] <= 0:
        b_val += scale * mu * vol_bottom * vx[x, y-1] # sv[2*x, 2*y-1, 0]
        
    # vy top +x
    if sphi[2*x+1, 2*y+2] <= 0:
        b_val += scale * mu * vol_top * vy[x,y+1] # sv[2*x+1, 2*y+2, 1]
    # vy top -x
    if sphi[2*x-1, 2*y+2] <= 0:
        b_val -= scale * mu * vol_top * vy[x-1,y+1] # sv[2*x-1, 2*y+2, 1]
    # vy bottom +x
    if sphi[2*x+1, 2*y] <= 0:
        b_val -= scale * mu * vol_bottom * vy[x,y] # sv[2*x+1, 2*y, 1]
    # vy bottom -x
    if sphi[2*x-1, 2*y] <= 0:
        b_val += scale * mu * vol_bottom * vy[x-1,y] # sv[2*x-1, 2*y, 1]
        
    
    b[x,y] = b_val

@cuda.jit
def initialize_solver_y_kernel(scale, mu, vx, vy, sphi, sv, vol, b):
    x, y = cuda.grid(2)
    if x == 0 or x >= b.shape[0] - 1 or y == 0 or y >= b.shape[1]-1 :
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y] <= 0:
        # ignore solid cells
        b[x,y] = 0
        return
    
    vol_center = vol[2*x+1, 2*y ]
    vol_right =  vol[2*x+2, 2*y ]
    vol_left =   vol[2*x,   2*y ]
    vol_top =    vol[2*x+1, 2*y+1]
    vol_bottom = vol[2*x+1, 2*y-1]

    b_val = vy[x,y] * vol_center
    
    # vy +x
    if sphi[2*x+3, 2*y] <= 0:
        b_val += scale * mu * vol_right * vy[x+1,y] # sv[2*x+3, 2*, 1]
    # vy -x
    if sphi[2*x-1, 2*y] <= 0:
        b_val += scale * mu * vol_left * vy[x-1,y] # sv[2*x-1, 2*y 1]
    # vy +y
    if sphi[2*x+1, 2*y+2] <= 0:
        b_val += 2 * scale * mu * vol_top * vy[x,y+1] # sv[2*x+1, 2*y+2 1]
    # vy -y
    if sphi[2*x+1, 2*y-2] <= 0:
        b_val += 2 * scale * mu * vol_bottom * vy[x,y-1] # sv[2*x+1, 2*y-2 1]
    
    # vx right +y
    if sphi[2*x+2, 2*y+1] <= 0:
        b_val += scale * mu * vol_right * vx[x+1,y] # sv[2*x+2, 2*y+1 0]
    # vx right -y
    if sphi[2*x+2, 2*y-1] <= 0:
        b_val -= scale * mu * vol_right * vx[x+1,y-1] # sv[2*x+2, 2*y-1 0]
    # vx left +y
    if sphi[2*x, 2*y+1] <= 0:
        b_val -= scale * mu * vol_left * vx[x,y] # sv[2*x, 2*y+1 0]
    # vx left -y
    if sphi[2*x, 2*y-1] <= 0:
        b_val += scale * mu * vol_left * vx[x,y-1] # sv[2*x, 2*y-1 0]
        
    b[x,y] = b_val


@cuda.jit
def matvecmul_x_kernel(scale, mu, vx, vy, out, sphi, vol):
    x, y = cuda.grid(2)
    if x == 0 or x >= out.shape[0] - 1 or y == 0 or y >= out.shape[1]-1 :
        # ignore boundary cells
        return

    if sphi[2*x, 2*y+1] <= 0:
        # ignore solid cells
        out[x,y] = 0
        return
    
    vol_center = vol[2*x,   2*y+1]
    vol_right =  vol[2*x+1, 2*y+1]
    vol_left =   vol[2*x-1, 2*y+1]
    vol_top =    vol[2*x,   2*y+2]
    vol_bottom = vol[2*x,   2*y ]
    vol_front =  vol[2*x,   2*y+1]
    vol_back =   vol[2*x,   2*y+1]
    
    diag = vol_center + scale * mu * (2*vol_right + 2*vol_left + vol_top + vol_bottom)
    val = diag * vx[x,y]
    
    # vx +x
    if sphi[2*x+2, 2*y+1] > 0:
        val -= 2 * scale * mu * vol_right * vx[x+1,y]
    # vx -x
    if sphi[2*x-2, 2*y+1] > 0:
        val -= 2 * scale * mu * vol_left * vx[x-1,y]
    # vx +y
    if sphi[2*x, 2*y+3] > 0:
        val -= scale * mu * vol_top * vx[x,y+1]
    # vx -y
    if sphi[2*x, 2*y-1] > 0:
        val -= scale * mu * vol_bottom * vx[x,y-1]
    
    # vy top +x
    if sphi[2*x+1, 2*y+2] > 0:
        val -= scale * mu * vol_top * vy[x,y+1]
    # vy top -x
    if sphi[2*x-1, 2*y+2] > 0:
        val += scale * mu * vol_top * vy[x-1,y+1]
    # vy bottom +x
    if sphi[2*x+1, 2*y] > 0:
        val += scale * mu * vol_bottom * vy[x,y]
    # vy bottom -x
    if sphi[2*x-1, 2*y] > 0:
        val -= scale * mu * vol_bottom * vy[x-1,y]
        

    out[x,y] = val

@cuda.jit
def matvecmul_y_kernel(scale, mu, vx, vy, out, sphi, vol):
    x, y = cuda.grid(2)
    if x == 0 or x >= out.shape[0] - 1 or y == 0 or y >= out.shape[1]-1 :
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y] <= 0:
        # ignore solid cells
        out[x,y] = 0
        return
    
    vol_center = vol[2*x+1, 2*y]
    vol_right =  vol[2*x+2, 2*y]
    vol_left =   vol[2*x,   2*y]
    vol_top =    vol[2*x+1, 2*y+1]
    vol_bottom = vol[2*x+1, 2*y-1]

    
    diag = vol_center + scale * mu * (vol_right + vol_left + 2*vol_top + 2*vol_bottom )
    val = diag * vy[x,y]
    
    # vy +x
    if sphi[2*x+3, 2*y] > 0:
        val -= scale * mu * vol_right * vy[x+1,y]
    # vy -x
    if sphi[2*x-1, 2*y] > 0:
        val -= scale * mu * vol_left * vy[x-1,y]
    # vy +y
    if sphi[2*x+1, 2*y+2] > 0:
        val -= 2 * scale * mu * vol_top * vy[x,y+1]
    # vy -y
    if sphi[2*x+1, 2*y-2] > 0:
        val -= 2 * scale * mu * vol_bottom * vy[x,y-1]
        
    # vx right +y
    if sphi[2*x+2, 2*y+1] > 0:
        val -= scale * mu * vol_right * vx[x+1,y]
    # vx right -y
    if sphi[2*x+2, 2*y-1] > 0:
        val += scale * mu * vol_right * vx[x+1,y-1]
    # vx left +y
    if sphi[2*x, 2*y+1] > 0:
        val += scale * mu * vol_left * vx[x,y]
    # vx left -y
    if sphi[2*x, 2*y-1] > 0:
        val -= scale * mu * vol_left * vx[x,y-1]
        
    
    out[x,y] = val


@cuda.jit
def apply_viscosity_kernel(gres, vx, vy, out_x, out_y, sphi, sv):
    x, y = cuda.grid(2)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1:
        # ignore boundary cells
        return
    
    if sphi[2*x, 2*y+1] > 0:
        vx[x,y] = out_x[x,y]
    if sphi[2*x+1, 2*y] > 0:
        vy[x,y] = out_y[x,y]

        
def initialize_solver(gres, scale, mu, vx, vy, sphi, sv, vol, b_x, b_y):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    gx_blocks = (*(((gres + cp.array([1,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gy_blocks = (*(((gres + cp.array([0,1], dtype=cp.int64)) - 1) // threads + 1).get(),)
  
    initialize_solver_x_kernel[gx_blocks, THREAD2](scale, mu, vx, vy,  sphi, sv, vol, b_x)
    initialize_solver_y_kernel[gy_blocks, THREAD2](scale, mu, vx, vy,  sphi, sv, vol, b_y)

def matvecmul(gres, scale, mu, vx, vy, out_x, out_y, sphi, vol):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    gx_blocks = (*(((gres + cp.array([1,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gy_blocks = (*(((gres + cp.array([0,1], dtype=cp.int64)) - 1) // threads + 1).get(),)
    
    matvecmul_x_kernel[gx_blocks, THREAD2](scale, mu, vx, vy, out_x, sphi, vol)
    matvecmul_y_kernel[gy_blocks, THREAD2](scale, mu, vx, vy, out_y, sphi, vol)

def apply_viscosity(gres, vx, vy, out_x, out_y,  sphi, sv):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    apply_viscosity_kernel[blocks, THREAD2](gres, vx, vy, out_x, out_y, sphi, sv)

class ViscosityCGSolver2D:
    def __init__(self, gres, bound_size):
        self.gres = gres
        self.cell_size = bound_size / gres
        self.cell_vol = cp.prod(self.cell_size).item()
        
        self.vol = cp.zeros((2*gres + 1).get(), dtype=cp.float64)
        
        self.d_x = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.d_y = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.r_x = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.r_y = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.q_x = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.q_y = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.x_x = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.x_y = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.b_x = cp.zeros((gres + cp.array([1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.b_y = cp.zeros((gres + cp.array([0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, dt, mu, rho, vx, vy, sphi, sv, lphi, lvol, tol=1e-4, save=False):
        scale = dt / self.cell_vol / rho
        
        self.vol[:] = lvol / (self.cell_vol * 0.125)
        self.x_x[:] = vx
        self.x_y[:] = vy
        
        initialize_solver(self.gres, scale, mu, self.x_x, self.x_y, sphi, sv, self.vol, self.b_x, self.b_y)
        matvecmul(self.gres, scale, mu, self.x_x, self.x_y,  self.q_x, self.q_y, sphi, self.vol)
        
        self.d_x[:] = self.b_x - self.q_x
        self.d_y[:] = self.b_y - self.q_y
        
        self.r_x[:] = self.d_x
        self.r_y[:] = self.d_y
        
        self.delta = (cp.sum(self.r_x ** 2) + cp.sum(self.r_y ** 2)).item()
        count=0
        if not self.delta < tol ** 2:
            for i in range(self.max_iter):
                matvecmul(self.gres, scale, mu, self.d_x, self.d_y, self.q_x, self.q_y, sphi, self.vol)
                cuda.synchronize()
                
                dq = (cp.sum(self.d_x * self.q_x) + cp.sum(self.d_y * self.q_y)).item()
                
                self.alpha = self.delta / dq
                self.x_x += self.alpha * self.d_x
                self.x_y += self.alpha * self.d_y
                
                self.r_x -= self.alpha * self.q_x
                self.r_y -= self.alpha * self.q_y

                old_delta = self.delta
                self.delta = (cp.sum(self.r_x ** 2) + cp.sum(self.r_y ** 2) ).item()
                if self.delta < tol ** 2:
                    break
                self.beta = self.delta / old_delta
                self.d_x[:] = self.r_x + self.beta * self.d_x
                self.d_y[:] = self.r_y + self.beta * self.d_y
                count+=1
            else:
                raise ValueError("Failed to converge!")
        #print(f'iteration is : {count}')
        apply_viscosity(self.gres, vx, vy, self.x_x, self.x_y,  sphi, sv)