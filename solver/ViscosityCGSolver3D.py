import math

import cupy as cp
from numba import cuda
import numpy as np
# from .SolidFraction2D import compute_solid_frac, edge_in_fraction

@cuda.jit
def extrapolate_kernel(v, valid, new_v, new_valid):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= v.shape[0] - 1 or y == 0 or y >= v.shape[1]-1 or z == 0 or z >= v.shape[2]-1:
        # ignore boundary cells
        return
    
    val = 0.0
    count = 0
    if valid[x,y,z]:
        return
    if valid[x+1,y,z]:
        val += v[x+1,y,z]
        count += 1
    if valid[x-1,y,z]:
        val += v[x-1,y,z]
        count += 1
    if valid[x,y+1,z]:
        val += v[x,y+1,z]
        count += 1
    if valid[x,y-1,z]:
        val += v[x,y-1,z]
        count += 1
    if valid[x,y,z+1]:
        val += v[x,y,z+1]
        count += 1
    if valid[x,y,z-1]:
        val += v[x,y,z-1]
        count += 1
    if count > 0:
        new_v[x,y,z] = val / count
        new_valid[x,y,z] = True

@cuda.jit
def initialize_solver_x_kernel(scale, mu, vx, vy, vz, sphi, sv, vol, b):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= b.shape[0] - 1 or y == 0 or y >= b.shape[1]-1 or z == 0 or z >= b.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x, 2*y+1, 2*z+1] < 0:
        # ignore solid cells
        b[x,y,z] = 0
        return
    
    vol_center = vol[2*x,   2*y+1, 2*z+1]
    vol_right =  vol[2*x+1, 2*y+1, 2*z+1]
    vol_left =   vol[2*x-1, 2*y+1, 2*z+1]
    vol_top =    vol[2*x,   2*y+2, 2*z+1]
    vol_bottom = vol[2*x,   2*y,   2*z+1]
    vol_front =  vol[2*x,   2*y+1, 2*z+2]
    vol_back =   vol[2*x,   2*y+1, 2*z]
    
    b_val = vx[x,y,z] * vol_center
    
    # vx +x
    if sphi[2*x+2, 2*y+1, 2*z+1] < 0:
        b_val += 2 * scale * mu * vol_right * vx[x+1, y, z] # sv[2*x+2, 2*y+1, 2*z+1, 0]
    # vx -x
    if sphi[2*x-2, 2*y+1, 2*z+1] < 0:
        b_val += 2 * scale * mu * vol_left * vx[x-1, y, z] # sv[2*x-2, 2*y+1, 2*z+1, 0]
    # vx +y
    if sphi[2*x, 2*y+3, 2*z+1] < 0:
        b_val += scale * mu * vol_top * vx[x, y+1, z] # sv[2*x, 2*y+3, 2*z+1, 0]
    # vx -y
    if sphi[2*x, 2*y-1, 2*z+1] < 0:
        b_val += scale * mu * vol_bottom * vx[x, y-1, z] # sv[2*x, 2*y-1, 2*z+1, 0]
    # vx +z
    if sphi[2*x, 2*y+1, 2*z+3] < 0:
        b_val += scale * mu * vol_front * vx[x, y, z+1] # sv[2*x, 2*y+1, 2*z+3, 0]
    # vx -z
    if sphi[2*x, 2*y+1, 2*z-1] < 0:
        b_val += scale * mu * vol_back * vx[x, y, z-1] # sv[2*x, 2*y+1, 2*z-1, 0]
    
    # vy top +x
    if sphi[2*x+1, 2*y+2, 2*z+1] < 0:
        b_val += scale * mu * vol_top * vy[x,y+1,z] # sv[2*x+1, 2*y+2, 2*z+1, 1]
    # vy top -x
    if sphi[2*x-1, 2*y+2, 2*z+1] < 0:
        b_val -= scale * mu * vol_top * vy[x-1,y+1,z] # sv[2*x-1, 2*y+2, 2*z+1, 1]
    # vy bottom +x
    if sphi[2*x+1, 2*y, 2*z+1] < 0:
        b_val -= scale * mu * vol_bottom * vy[x,y,z] # sv[2*x+1, 2*y, 2*z+1, 1]
    # vy bottom -x
    if sphi[2*x-1, 2*y, 2*z+1] < 0:
        b_val += scale * mu * vol_bottom * vy[x-1,y,z] # sv[2*x-1, 2*y, 2*z+1, 1]
        
    # vz front +x
    if sphi[2*x+1, 2*y+1, 2*z+2] < 0:
        b_val += scale * mu * vol_front * vz[x,y,z+1] # sv[2*x+1, 2*y+1, 2*z+2, 2]
    # vz front -x
    if sphi[2*x-1, 2*y+1, 2*z+2] < 0:
        b_val -= scale * mu * vol_front * vz[x-1,y,z+1] # sv[2*x-1, 2*y+1, 2*z+2, 2]
    # vz back +x
    if sphi[2*x+1, 2*y+1, 2*z] < 0:
        b_val -= scale * mu * vol_back * vz[x,y,z] # sv[2*x+1, 2*y+1, 2*z, 2]
    # vz back -x
    if sphi[2*x-1, 2*y+1, 2*z] < 0:
        b_val += scale * mu * vol_back * vz[x-1,y,z] # sv[2*x-1, 2*y+1, 2*z, 2]
    
    b[x,y,z] = b_val

@cuda.jit
def initialize_solver_y_kernel(scale, mu, vx, vy, vz, sphi, sv, vol, b):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= b.shape[0] - 1 or y == 0 or y >= b.shape[1]-1 or z == 0 or z >= b.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y, 2*z+1] < 0:
        # ignore solid cells
        b[x,y,z] = 0
        return
    
    vol_center = vol[2*x+1, 2*y,   2*z+1]
    vol_right =  vol[2*x+2, 2*y,   2*z+1]
    vol_left =   vol[2*x,   2*y,   2*z+1]
    vol_top =    vol[2*x+1, 2*y+1, 2*z+1]
    vol_bottom = vol[2*x+1, 2*y-1, 2*z+1]
    vol_front =  vol[2*x+1, 2*y,   2*z+2]
    vol_back =   vol[2*x+1, 2*y,   2*z]

    b_val = vy[x,y,z] * vol_center
    
    # vy +x
    if sphi[2*x+3, 2*y, 2*z+1] < 0:
        b_val += scale * mu * vol_right * vy[x+1,y,z] # sv[2*x+3, 2*y, 2*z+1, 1]
    # vy -x
    if sphi[2*x-1, 2*y, 2*z+1] < 0:
        b_val += scale * mu * vol_left * vy[x-1,y,z] # sv[2*x-1, 2*y, 2*z+1, 1]
    # vy +y
    if sphi[2*x+1, 2*y+2, 2*z+1] < 0:
        b_val += 2 * scale * mu * vol_top * vy[x,y+1,z] # sv[2*x+1, 2*y+2, 2*z+1, 1]
    # vy -y
    if sphi[2*x+1, 2*y-2, 2*z+1] < 0:
        b_val += 2 * scale * mu * vol_bottom * vy[x,y-1,z] # sv[2*x+1, 2*y-2, 2*z+1, 1]
    # vy +z
    if sphi[2*x+1, 2*y, 2*z+3] < 0:
        b_val += scale * mu * vol_front * vy[x,y,z+1] # sv[2*x+1, 2*y, 2*z+3, 1]
    # vy -z
    if sphi[2*x+1, 2*y, 2*z-1] < 0:
        b_val += scale * mu * vol_back * vy[x,y,z-1] # sv[2*x+1, 2*y, 2*z-1, 1]
    
    # vx right +y
    if sphi[2*x+2, 2*y+1, 2*z+1] < 0:
        b_val += scale * mu * vol_right * vx[x+1,y,z] # sv[2*x+2, 2*y+1, 2*z+1, 0]
    # vx right -y
    if sphi[2*x+2, 2*y-1, 2*z+1] < 0:
        b_val -= scale * mu * vol_right * vx[x+1,y-1,z] # sv[2*x+2, 2*y-1, 2*z+1, 0]
    # vx left +y
    if sphi[2*x, 2*y+1, 2*z+1] < 0:
        b_val -= scale * mu * vol_left * vx[x,y,z] # sv[2*x, 2*y+1, 2*z+1, 0]
    # vx left -y
    if sphi[2*x, 2*y-1, 2*z+1] < 0:
        b_val += scale * mu * vol_left * vx[x,y-1,z] # sv[2*x, 2*y-1, 2*z+1, 0]
        
    # vz front +y
    if sphi[2*x+1, 2*y+1, 2*z+2] < 0:
        b_val += scale * mu * vol_front * vz[x,y,z+1] # sv[2*x+1, 2*y+1, 2*z+2, 2]
    # vz front -y
    if sphi[2*x+1, 2*y-1, 2*z+2] < 0:
        b_val -= scale * mu * vol_front * vz[x,y-1,z+1] # sv[2*x+1, 2*y-1, 2*z+2, 2]
    # vz back +y
    if sphi[2*x+1, 2*y+1, 2*z] < 0:
        b_val -= scale * mu * vol_back * vz[x,y,z] # sv[2*x+1, 2*y+1, 2*z, 2]
    # vz back -y
    if sphi[2*x+1, 2*y-1, 2*z] < 0:
        b_val += scale * mu * vol_back * vz[x,y-1,z] # sv[2*x+1, 2*y-1, 2*z, 2]
    
    b[x,y,z] = b_val

@cuda.jit
def initialize_solver_z_kernel(scale, mu, vx, vy, vz, sphi, sv, vol, b):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= b.shape[0] - 1 or y == 0 or y >= b.shape[1]-1 or z == 0 or z >= b.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y+1, 2*z] < 0:
        # ignore solid cells
        b[x,y,z] = 0
        return
    
    vol_center = vol[2*x+1, 2*y+1, 2*z]
    vol_right =  vol[2*x+2, 2*y+1, 2*z]
    vol_left =   vol[2*x,   2*y+1, 2*z]
    vol_top =    vol[2*x+1, 2*y+2, 2*z]
    vol_bottom = vol[2*x+1, 2*y,   2*z]
    vol_front =  vol[2*x+1, 2*y+1, 2*z+1]
    vol_back =   vol[2*x+1, 2*y+1, 2*z-1]

    b_val = vz[x,y,z] * vol_center
    
    # vz +x
    if sphi[2*x+3, 2*y+1, 2*z] < 0:
        b_val += scale * mu * vol_right * vz[x+1,y,z] # sv[2*x+3, 2*y+1, 2*z, 2]
    # vz -x
    if sphi[2*x-1, 2*y+1, 2*z] < 0:
        b_val += scale * mu * vol_left * vz[x-1,y,z] # sv[2*x-1, 2*y+1, 2*z, 2]
    # vz +y
    if sphi[2*x+1, 2*y+3, 2*z] < 0:
        b_val += scale * mu * vol_top * vz[x,y+1,z] # sv[2*x+1, 2*y+3, 2*z, 2]
    # vz -y
    if sphi[2*x+1, 2*y-1, 2*z] < 0:
        b_val += scale * mu * vol_bottom * vz[x,y-1,z] # sv[2*x+1, 2*y-1, 2*z, 2]
    # vz +z
    if sphi[2*x+1, 2*y+1, 2*z+2] < 0:
        b_val += 2 * scale * mu * vol_front * vz[x,y,z+1] # sv[2*x+1, 2*y+1, 2*z+2, 2]
    # vy -z
    if sphi[2*x+1, 2*y+1, 2*z-2] < 0:
        b_val += 2 * scale * mu * vol_back * vz[x,y,z-1] # sv[2*x+1, 2*y+1, 2*z-2, 2]
    
    # vx right +z
    if sphi[2*x+2, 2*y+1, 2*z+1] < 0:
        b_val += scale * mu * vol_right * vx[x+1,y,z] # sv[2*x+2, 2*y+1, 2*z+1, 0]
    # vx right -z
    if sphi[2*x+2, 2*y+1, 2*z-1] < 0:
        b_val -= scale * mu * vol_right * vx[x+1,y,z-1] # sv[2*x+2, 2*y+1, 2*z-1, 0]
    # vx left +z
    if sphi[2*x, 2*y+1, 2*z+1] < 0:
        b_val -= scale * mu * vol_left * vx[x,y,z] # sv[2*x, 2*y+1, 2*z+1, 0]
    # vx left -z
    if sphi[2*x, 2*y+1, 2*z-1] < 0:
        b_val += scale * mu * vol_left * vx[x,y,z-1] # sv[2*x, 2*y+1, 2*z-1, 0]
        
    # vy top +z
    if sphi[2*x+1, 2*y+2, 2*z+1] < 0:
        b_val += scale * mu * vol_top * vy[x,y+1,z] # sv[2*x+1, 2*y+2, 2*z+1, 1]
    # vy top -z
    if sphi[2*x+1, 2*y+2, 2*z-1] < 0:
        b_val -= scale * mu * vol_top * vy[x,y+1,z-1] # sv[2*x+1, 2*y+2, 2*z-1, 1]
    # vy bottom +z
    if sphi[2*x+1, 2*y, 2*z+1] < 0:
        b_val -= scale * mu * vol_bottom * vy[x,y,z] # sv[2*x+1, 2*y, 2*z+1, 1]
    # vy bottom -z
    if sphi[2*x+1, 2*y, 2*z-1] < 0:
        b_val += scale * mu * vol_bottom * vy[x,y,z-1] # sv[2*x+1, 2*y, 2*z-1, 1]
    
    b[x,y,z] = b_val

@cuda.jit
def matvecmul_x_kernel(scale, mu, vx, vy, vz, out, sphi, vol):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= out.shape[0] - 1 or y == 0 or y >= out.shape[1]-1 or z == 0 or z >= out.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x, 2*y+1, 2*z+1] < 0:
        # ignore solid cells
        out[x,y,z] = 0
        return
    
    vol_center = vol[2*x,   2*y+1, 2*z+1]
    vol_right =  vol[2*x+1, 2*y+1, 2*z+1]
    vol_left =   vol[2*x-1, 2*y+1, 2*z+1]
    vol_top =    vol[2*x,   2*y+2, 2*z+1]
    vol_bottom = vol[2*x,   2*y,   2*z+1]
    vol_front =  vol[2*x,   2*y+1, 2*z+2]
    vol_back =   vol[2*x,   2*y+1, 2*z]
    
    diag = vol_center + scale * mu * (2*vol_right + 2*vol_left + vol_top + vol_bottom + vol_front + vol_back)
    val = diag * vx[x,y,z]
    
    # vx +x
    if sphi[2*x+2, 2*y+1, 2*z+1] >= 0:
        val -= 2 * scale * mu * vol_right * vx[x+1,y,z]
    # vx -x
    if sphi[2*x-2, 2*y+1, 2*z+1] >= 0:
        val -= 2 * scale * mu * vol_left * vx[x-1,y,z]
    # vx +y
    if sphi[2*x, 2*y+3, 2*z+1] >= 0:
        val -= scale * mu * vol_top * vx[x,y+1,z]
    # vx -y
    if sphi[2*x, 2*y-1, 2*z+1] >= 0:
        val -= scale * mu * vol_bottom * vx[x,y-1,z]
    # vx +z
    if sphi[2*x, 2*y+1, 2*z+3] >= 0:
        val -= scale * mu * vol_front * vx[x,y,z+1]
    # vx -z
    if sphi[2*x, 2*y+1, 2*z-1] >= 0:
        val -= scale * mu * vol_back * vx[x,y,z-1]
    
    # vy top +x
    if sphi[2*x+1, 2*y+2, 2*z+1] >= 0:
        val -= scale * mu * vol_top * vy[x,y+1,z]
    # vy top -x
    if sphi[2*x-1, 2*y+2, 2*z+1] >= 0:
        val += scale * mu * vol_top * vy[x-1,y+1,z]
    # vy bottom +x
    if sphi[2*x+1, 2*y, 2*z+1] >= 0:
        val += scale * mu * vol_bottom * vy[x,y,z]
    # vy bottom -x
    if sphi[2*x-1, 2*y, 2*z+1] >= 0:
        val -= scale * mu * vol_bottom * vy[x-1,y,z]
        
    # vz front +x
    if sphi[2*x+1, 2*y+1, 2*z+2] >= 0:
        val -= scale * mu * vol_front * vz[x,y,z+1]
    # vz front -x
    if sphi[2*x-1, 2*y+1, 2*z+2] >= 0:
        val += scale * mu * vol_front * vz[x-1,y,z+1]
    # vz back +x
    if sphi[2*x+1, 2*y+1, 2*z] >= 0:
        val += scale * mu * vol_back * vz[x,y,z]
    # vz back -x
    if sphi[2*x-1, 2*y+1, 2*z] >= 0:
        val -= scale * mu * vol_back * vz[x-1,y,z]
    
    out[x,y,z] = val

@cuda.jit
def matvecmul_y_kernel(scale, mu, vx, vy, vz, out, sphi, vol):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= out.shape[0] - 1 or y == 0 or y >= out.shape[1]-1 or z == 0 or z >= out.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y, 2*z+1] < 0:
        # ignore solid cells
        out[x,y,z] = 0
        return
    
    vol_center = vol[2*x+1, 2*y,   2*z+1]
    vol_right =  vol[2*x+2, 2*y,   2*z+1]
    vol_left =   vol[2*x,   2*y,   2*z+1]
    vol_top =    vol[2*x+1, 2*y+1, 2*z+1]
    vol_bottom = vol[2*x+1, 2*y-1, 2*z+1]
    vol_front =  vol[2*x+1, 2*y,   2*z+2]
    vol_back =   vol[2*x+1, 2*y,   2*z]
    
    diag = vol_center + scale * mu * (vol_right + vol_left + 2*vol_top + 2*vol_bottom + vol_front + vol_back)
    val = diag * vy[x,y,z]
    
    # vy +x
    if sphi[2*x+3, 2*y, 2*z+1] >= 0:
        val -= scale * mu * vol_right * vy[x+1,y,z]
    # vy -x
    if sphi[2*x-1, 2*y, 2*z+1] >= 0:
        val -= scale * mu * vol_left * vy[x-1,y,z]
    # vy +y
    if sphi[2*x+1, 2*y+2, 2*z+1] >= 0:
        val -= 2 * scale * mu * vol_top * vy[x,y+1,z]
    # vy -y
    if sphi[2*x+1, 2*y-2, 2*z+1] >= 0:
        val -= 2 * scale * mu * vol_bottom * vy[x,y-1,z]
    # vy +z
    if sphi[2*x+1, 2*y, 2*z+3] >= 0:
        val -= scale * mu * vol_front * vy[x,y,z+1]
    # vy -z
    if sphi[2*x+1, 2*y, 2*z-1] >= 0:
        val -= scale * mu * vol_back * vy[x,y,z-1]
    
    # vx right +y
    if sphi[2*x+2, 2*y+1, 2*z+1] >= 0:
        val -= scale * mu * vol_right * vx[x+1,y,z]
    # vx right -y
    if sphi[2*x+2, 2*y-1, 2*z+1] >= 0:
        val += scale * mu * vol_right * vx[x+1,y-1,z]
    # vx left +y
    if sphi[2*x, 2*y+1, 2*z+1] >= 0:
        val += scale * mu * vol_left * vx[x,y,z]
    # vx left -y
    if sphi[2*x, 2*y-1, 2*z+1] >= 0:
        val -= scale * mu * vol_left * vx[x,y-1,z]
        
    # vz front +y
    if sphi[2*x+1, 2*y+1, 2*z+2] >= 0:
        val -= scale * mu * vol_front * vz[x,y,z+1]
    # vz front -y
    if sphi[2*x+1, 2*y-1, 2*z+2] >= 0:
        val += scale * mu * vol_front * vz[x,y-1,z+1]
    # vz back +y
    if sphi[2*x+1, 2*y+1, 2*z] >= 0:
        val += scale * mu * vol_back * vz[x,y,z]
    # vz back -y
    if sphi[2*x+1, 2*y-1, 2*z] >= 0:
        val -= scale * mu * vol_back * vz[x,y-1,z]
    
    out[x,y,z] = val

@cuda.jit
def matvecmul_z_kernel(scale, mu, vx, vy, vz, out, sphi, vol):
    x, y, z = cuda.grid(3)
    if x == 0 or x >= out.shape[0] - 1 or y == 0 or y >= out.shape[1]-1 or z == 0 or z >= out.shape[2]-1:
        # ignore boundary cells
        return

    if sphi[2*x+1, 2*y+1, 2*z] < 0:
        # ignore solid cells
        out[x,y,z] = 0
        return
    
    vol_center = vol[2*x+1, 2*y+1, 2*z]
    vol_right =  vol[2*x+2, 2*y+1, 2*z]
    vol_left =   vol[2*x,   2*y+1, 2*z]
    vol_top =    vol[2*x+1, 2*y+2, 2*z]
    vol_bottom = vol[2*x+1, 2*y,   2*z]
    vol_front =  vol[2*x+1, 2*y+1, 2*z+1]
    vol_back =   vol[2*x+1, 2*y+1, 2*z-1]
    
    diag = vol_center + scale * mu * (vol_right + vol_left + vol_top + vol_bottom + 2*vol_front + 2*vol_back)
    val = diag * vz[x,y,z]
    
    # vz +x
    if sphi[2*x+3, 2*y+1, 2*z] >= 0:
        val -= scale * mu * vol_right * vz[x+1,y,z]
    # vz -x
    if sphi[2*x-1, 2*y+1, 2*z] >= 0:
        val -= scale * mu * vol_left * vz[x-1,y,z]
    # vz +y
    if sphi[2*x+1, 2*y+3, 2*z] >= 0:
        val -= scale * mu * vol_top * vz[x,y+1,z]
    # vz -y
    if sphi[2*x+1, 2*y-1, 2*z] >= 0:
        val -= scale * mu * vol_bottom * vz[x,y-1,z]
    # vz +z
    if sphi[2*x+1, 2*y+1, 2*z+2] >= 0:
        val -= 2 * scale * mu * vol_front * vz[x,y,z+1]
    # vz -z
    if sphi[2*x+1, 2*y+1, 2*z-2] >= 0:
        val -= 2 * scale * mu * vol_back * vz[x,y,z-1]
    
    # vx right +z
    if sphi[2*x+2, 2*y+1, 2*z+1] >= 0:
        val -= scale * mu * vol_right * vx[x+1,y,z]
    # vx right -z
    if sphi[2*x+2, 2*y+1, 2*z-1] >= 0:
        val += scale * mu * vol_right * vx[x+1,y,z-1]
    # vx left +z
    if sphi[2*x, 2*y+1, 2*z+1] >= 0:
        val += scale * mu * vol_left * vx[x,y,z]
    # vx left -z
    if sphi[2*x, 2*y+1, 2*z-1] >= 0:
        val -= scale * mu * vol_left * vx[x,y,z-1]
        
    # vy top +z
    if sphi[2*x+1, 2*y+2, 2*z+1] >= 0:
        val -= scale * mu * vol_top * vy[x,y+1,z]
    # vy top -z
    if sphi[2*x+1, 2*y+2, 2*z-1] >= 0:
        val += scale * mu * vol_top * vy[x,y+1,z-1]
    # vy bottom +z
    if sphi[2*x+1, 2*y, 2*z+1] >= 0:
        val += scale * mu * vol_bottom * vy[x,y,z]
    # vy bottom -z
    if sphi[2*x+1, 2*y, 2*z-1] >= 0:
        val -= scale * mu * vol_bottom * vy[x,y,z-1]
    
    out[x,y,z] = val

@cuda.jit
def apply_viscosity_kernel(gres, vx, vy, vz, out_x, out_y, out_z, sphi, sv):
    x, y, z = cuda.grid(3)
    if x == 0 or x > gres[0] - 1 or y == 0 or y > gres[1]-1 or z == 0 or z > gres[2]-1:
        # ignore boundary cells
        return
    
    if sphi[2*x, 2*y+1, 2*z+1] >= 0:
        vx[x,y,z] = out_x[x,y,z]
    if sphi[2*x+1, 2*y, 2*z+1] >= 0:
        vy[x,y,z] = out_y[x,y,z]
    if sphi[2*x+1, 2*y+1, 2*z] >= 0:
        vz[x,y,z] = out_z[x,y,z]

def extrapolate(gres, num_iter, vx, vy, vz, sphi):
    THREAD3 = (8, 8, 8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    gx_blocks = (*(((gres + cp.array([1,0,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gy_blocks = (*(((gres + cp.array([0,1,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gz_blocks = (*(((gres + cp.array([0,0,1], dtype=cp.int64)) - 1) // threads + 1).get(),)
    
    vx_valid = sphi[0::2, 1::2, 1::2] >= 0
    vy_valid = sphi[1::2, 0::2, 1::2] >= 0
    vz_valid = sphi[1::2, 1::2, 0::2] >= 0
    
    new_vx = vx.copy()
    new_vy = vy.copy()
    new_vz = vz.copy()
    
    new_vx_valid = vx_valid.copy()
    new_vy_valid = vy_valid.copy()
    new_vz_valid = vz_valid.copy()
    
    for _ in range(num_iter):
        extrapolate_kernel[gx_blocks, THREAD3](vx, vx_valid, new_vx, new_vx_valid)
        extrapolate_kernel[gy_blocks, THREAD3](vy, vy_valid, new_vy, new_vy_valid)
        extrapolate_kernel[gx_blocks, THREAD3](vz, vz_valid, new_vz, new_vz_valid)
        
        vx[:] = new_vx
        vy[:] = new_vy
        vz[:] = new_vz
        
        vx_valid[:] = new_vx_valid
        vy_valid[:] = new_vy_valid
        vz_valid[:] = new_vz_valid
        
def initialize_solver(gres, scale, mu, vx, vy, vz, sphi, sv, vol, b_x, b_y, b_z):
    THREAD3 = (8, 8, 8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    gx_blocks = (*(((gres + cp.array([1,0,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gy_blocks = (*(((gres + cp.array([0,1,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gz_blocks = (*(((gres + cp.array([0,0,1], dtype=cp.int64)) - 1) // threads + 1).get(),)
    
    initialize_solver_x_kernel[gx_blocks, THREAD3](scale, mu, vx, vy, vz, sphi, sv, vol, b_x)
    initialize_solver_y_kernel[gy_blocks, THREAD3](scale, mu, vx, vy, vz, sphi, sv, vol, b_y)
    initialize_solver_z_kernel[gz_blocks, THREAD3](scale, mu, vx, vy, vz, sphi, sv, vol, b_z)

def matvecmul(gres, scale, mu, vx, vy, vz, out_x, out_y, out_z, sphi, vol):
    THREAD3 = (8, 8, 8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    gx_blocks = (*(((gres + cp.array([1,0,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gy_blocks = (*(((gres + cp.array([0,1,0], dtype=cp.int64)) - 1) // threads + 1).get(),)
    gz_blocks = (*(((gres + cp.array([0,0,1], dtype=cp.int64)) - 1) // threads + 1).get(),)
    
    matvecmul_x_kernel[gx_blocks, THREAD3](scale, mu, vx, vy, vz, out_x, sphi, vol)
    matvecmul_y_kernel[gy_blocks, THREAD3](scale, mu, vx, vy, vz, out_y, sphi, vol)
    matvecmul_z_kernel[gz_blocks, THREAD3](scale, mu, vx, vy, vz, out_z, sphi, vol)

def apply_viscosity(gres, vx, vy, vz, out_x, out_y, out_z, sphi, sv):
    THREAD3 = (8, 8, 8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    apply_viscosity_kernel[blocks, THREAD3](gres, vx, vy, vz, out_x, out_y, out_z, sphi, sv)

class ViscosityCGSolver3D:
    def __init__(self, gres, bound_size):
        self.gres = gres
        self.cell_size = bound_size / gres
        self.cell_vol = cp.prod(self.cell_size).item()
        
        self.vol = cp.zeros((2*gres + 1).get(), dtype=cp.float64)
        
        self.d_x = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.d_y = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.d_z = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.r_x = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.r_y = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.r_z = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.q_x = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.q_y = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.q_z = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.x_x = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.x_y = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.x_z = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.b_x = cp.zeros((gres + cp.array([1,0,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.b_y = cp.zeros((gres + cp.array([0,1,0], dtype=cp.int64)).get(), dtype=cp.float64)
        self.b_z = cp.zeros((gres + cp.array([0,0,1], dtype=cp.int64)).get(), dtype=cp.float64)
        
        self.alpha = 0.0
        self.beta = 0.0
        self.delta = 0.0
        
        self.max_iter = cp.prod(self.gres).item()
    
    def solve(self, dt, mu, rho, vx, vy, vz, sphi, sv, lphi, lvol, tol=1e-3):
        scale = dt / self.cell_vol / rho
        self.vol[:] = lvol / (self.cell_vol * 0.125)
        self.x_x[:] = vx
        self.x_y[:] = vy
        self.x_z[:] = vz
        
        extrapolate(self.gres, 3, self.x_x, self.x_y, self.x_z, sphi)
        initialize_solver(self.gres, scale, mu, self.x_x, self.x_y, self.x_z, sphi, sv, self.vol, self.b_x, self.b_y, self.b_z)
        matvecmul(self.gres, scale, mu, self.x_x, self.x_y, self.x_z, self.q_x, self.q_y, self.q_z, sphi, self.vol)
        
        self.d_x[:] = self.b_x - self.q_x
        self.d_y[:] = self.b_y - self.q_y
        self.d_z[:] = self.b_z - self.q_z
        
        self.r_x[:] = self.d_x
        self.r_y[:] = self.d_y
        self.r_z[:] = self.d_z
        
        self.delta = (cp.sum(self.r_x ** 2) + cp.sum(self.r_y ** 2) + cp.sum(self.r_z ** 2)).item()
        
        if not self.delta < tol ** 2:
            for i in range(self.max_iter):
                matvecmul(self.gres, scale, mu, self.d_x, self.d_y, self.d_z, self.q_x, self.q_y, self.q_z, sphi, self.vol)
                cuda.synchronize()
                
                dq = (cp.sum(self.d_x * self.q_x) + cp.sum(self.d_y * self.q_y) + cp.sum(self.d_z * self.q_z)).item()
                
                self.alpha = self.delta / dq
                self.x_x += self.alpha * self.d_x
                self.x_y += self.alpha * self.d_y
                self.x_z += self.alpha * self.d_z
                
                self.r_x -= self.alpha * self.q_x
                self.r_y -= self.alpha * self.q_y
                self.r_z -= self.alpha * self.q_z

                old_delta = self.delta
                self.delta = (cp.sum(self.r_x ** 2) + cp.sum(self.r_y ** 2) + cp.sum(self.r_z ** 2)).item()
                if self.delta < tol ** 2:
                    break
                self.beta = self.delta / old_delta
                self.d_x[:] = self.r_x + self.beta * self.d_x
                self.d_y[:] = self.r_y + self.beta * self.d_y
                self.d_z[:] = self.r_z + self.beta * self.d_z
            else:
                raise ValueError("Failed to converge!")
        apply_viscosity(self.gres, vx, vy, vz, self.x_x, self.x_y, self.x_z, sphi, sv)
        
