# from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R
import cupy as cp
import math
# from common import *
# from utils import *
# from sparse_solver import solve
import numpy as np
from numba import cuda
from matplotlib import pyplot as plt

@cuda.jit
def mat_TR(T,R,TR):
    for i in range(2):
        for j in range(2):
            TR[i,j]=R[i,j]
        TR[i,2] = T[i,2]

@cuda.jit
def matvecmul4(A, B, C):
    """Perform square matrix multiplication of C = A * B in 4by4 matrix 3 vec C: 3 vec"""  
    for i in range(2):
        tmp = 0.
        for j in range(2):
            tmp += A[i,j]*B[j]
        tmp += A[i,2]
        C[i]=tmp

@cuda.jit
def inv_rigid(T, R, inv_tr):
    """inv_rigid in 4*4 size"""  
    for i in range(2):
        tmp = 0.
        for j in range(2):
            inv_tr[i,j]=R[j,i]
            tmp -= R[j,i]*T[j, 2]
        inv_tr[i, 2] = tmp
    inv_tr[2,2]=1

@cuda.jit
def norm(pos):
    return (pos[0]**2+pos[1]**2)**0.5

@cuda.jit
def vec2_norm(pos, norm_pos):
    norm = (pos[0]**2+pos[1]**2)**0.5
    norm_pos[0] = pos[0]/norm
    norm_pos[1] = pos[1]/norm
    
@cuda.jit
def sphere_eval(rb, position):
    disp = cuda.local.array(2, dtype=cp.float64)
    
    disp[0] = position[0] - rb[1,2]
    disp[1] = position[1] - rb[2,2]
    dist = norm(disp)
    disp_normalized = cuda.local.array(2, dtype=cp.float64)
    vec2_norm(disp,disp_normalized)
    sd = dist - rb[0, 1] # rb[0,1]: radius
    if rb[0, 0]%2: #flipped
        sd = -sd
    return sd
    
@cuda.jit
def sphere_project(rb, position):
    #sd, disp_normalized = sphere_eval(rb, position)
    disp = cuda.local.array(2, dtype=cp.float64)
    disp[0] = position[0] - rb[1,2]
    disp[1] = position[1] - rb[2,2]
    dist = norm(disp)
    if dist <=0.0001:
        if rb[0,0]%2: #inside
            position[0] = rb[1,2]+rb[0,1]
            position[1] = rb[2,2]
    else:
        disp_normalized = cuda.local.array(2, dtype=cp.float64)
        vec2_norm(disp, disp_normalized)
        sd = dist - rb[0, 1] # rb[0,1]: radius
        if rb[0, 0]%2: #flipped
            sd = -sd
        if sd < 0:
            for i in range(2):
                position[i] = disp_normalized[i] * rb[0,1] + rb[i+1,2]

@cuda.jit
def box_eval(rb, position):
    
    inv_TR = cuda.local.array((3,3), dtype=cp.float64)
    inv_rigid(rb[1:4,:], rb[4:,:], inv_TR)
    pos_rb = cuda.local.array(2, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    
    disp = cuda.local.array(2, dtype=cp.float64)
    
    for i in range(2):
        disp[i] = abs(pos_rb[i]) - rb[0, 1+i]/2
    
    tmp=0.0
    max_disp=-100
    for i in range(2):
        if disp[i]>0: tmp += disp[i]**2
        if max_disp < disp[i]: max_disp = disp[i]
    sd = tmp ** 0.5 
    if max_disp < 0: sd += max_disp
    
    if rb[0, 0] % 2: #flipped
        sd = -sd
    return sd
    
@cuda.jit
def box_project(rb, position):
    inv_TR = cuda.local.array((3,3), dtype=cp.float64)
    inv_rigid(rb[1:4,:], rb[4:7,:], inv_TR)
    pos_rb = cuda.local.array(2, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    
    in_out = 0 # 0 means in 
    for i in range(2):
        if pos_rb[i]> rb[0, 1+i]/2 or pos_rb[i]< -rb[0, 1+i]/2:
            in_out +=1

    if rb[0, 0] % 2 and ~(in_out): # flipped, and in
        for i in range(2): #pos_rb[:3] = cp.clip(pos_rb[:3],-half_size, half_size)
            if pos_rb[i]<-rb[0, 1+i]/2: pos_rb[i] = -rb[0, 1+i]/2
            elif pos_rb[i]>rb[0, 1+i]/2: pos_rb[i] = rb[0, 1+i]/2
        TR = cuda.local.array((3,3), dtype=cp.float64)
        mat_TR(rb[1:4,:], rb[4:7,:], TR)
        matvecmul4(TR, pos_rb, position)

    elif in_out==0: # not flipped, and in
        index=0 # 0:-x, 1:+x, 2:-y, 3:+y, 4:-z, 5:+z
        dist_xyz=100
        for i in range(2):
            if rb[0,1+i]/2 - pos_rb[i] < dist_xyz:
                dist_xyz = rb[0,1+i]/2 - pos_rb[i]
                index = i*2
            if pos_rb[i]+ rb[0,1+i]/2 < dist_xyz:
                dist_xyz = pos_rb[i]+ rb[0,1+i]/2
                index = i*2+1
        pos_rb[index//2] += dist_xyz * (-1)**(index%2)
        TR = cuda.local.array((3,3), dtype=cp.float64)
        mat_TR(rb[1:4,:], rb[4:7,:], TR)
        matvecmul4(TR, pos_rb, position)


@cuda.jit
def evaluate_kernel(rb_d, sd, vel, position):
    # rb_d: all rb information n*9*4
    # sd: signed distance for position
    # 3d position must be in cp type
    P = cuda.grid(1)
    if P >= position.shape[0]:
        return
    
    min_sd = 100
    rb_index = 0
    for i in range(rb_d.shape[0]):
        if rb_d[i,0,0]//2 == 0: #sphere
            d = sphere_eval(rb_d[i,:,:], position[P,:])
        elif rb_d[i,0,0]//2 == 1: #box
            d = box_eval(rb_d[i,:,:], position[P,:])

        if d < min_sd: 
            min_sd = d
            rb_index = i
    sd[P] = min_sd
    if min_sd <= 0:
        for i in range(2):
            vel[P,i] = rb_d[rb_index, -1, i]
    
@cuda.jit
def project_kernel(rb_d, position):
    # rb_d: all rb information n*9*4
    # 3d position must be in cp type
    P = cuda.grid(1)
    if P >= position.shape[0]:
        return
    
    for i in range(rb_d.shape[0]):
        if rb_d[i,0,0]//2 == 0: #sphere
            sphere_project(rb_d[i,:,:], position[P,:])
        elif rb_d[i, 0, 0]//2 == 1: #box
            box_project(rb_d[i,:,:], position[P,:])
            
def evaluate(rb_d, sd, vel, position):
    assert sd.shape == position.shape[:-1]
    assert position.shape[-1] == 2
    assert vel.shape[-1] == 2
    
    vel *= 0
    NUM_POSITIONS = position.reshape(-1,2).shape[0]
    
    THREAD1 = 256
    blocks = ((NUM_POSITIONS-1) // THREAD1) + 1
    evaluate_kernel[blocks, THREAD1](rb_d, sd.reshape(-1), vel.reshape(-1,2), position.reshape(-1,2))
    cuda.synchronize()

def project(rb_d, position):
    assert position.shape[-1] == 2
    
    THREAD1 = 256
    blocks = ((position.shape[0]-1) // THREAD1) + 1

    project_kernel[blocks, THREAD1](rb_d, position)
    cuda.synchronize()
    
def get_T(position):
    t_mat = cp.identity((3))
    t_mat[0:2,2] = cp.asarray(position)
    return t_mat

def get_R(axis, angle):
    r_mat = cp.identity((3))
    if angle:
        rad = angle * np.pi /180
        c,s = np.cos(rad), np.sin(rad)
        r_mat[:2,:2]=cp.asarray(((c, -s), (s, c)))
    return r_mat
    

def generate_rb(rb_d, rb_map, name, rbparam, flip=False, center=[0, 0], axis=[0, 1], angle=0):
    # arguments
    # rb_d: all rb information
    # rbparam: ['sphere', radius], ['box', x_scale, y_scale]
    # rb_d: number of rb * 9 * 4 matrix
    # one rb has size 9 by 4 matrix
    # row 0: rbparam 
        # sphere: [0, radius, 0, 0], 1 means flipped
        # box: [2, x_scale, y_scale, z_scale], 3 means flipped
        # cylinder: [4, radius, height, 0] center at origin, 5 means flipped
    # row 1-3: translation matrix in mat3
    # row 4-6: rotation axis and angle in mat3
    # row 7: set solid velocity
    
    rb = cp.zeros((1, 8, 3))
    if rbparam[0]=='sphere':
        rb[:, 0, 0] = 1 if flip else 0
        rb[:, 0, 1] = rbparam[1]
    elif rbparam[0]=='box':
        rb[:, 0, 0] = 3 if flip else 2
        rb[:, 0, 1:] = cp.asarray(rbparam[1:])
    else:
        return rb_d
    
    rb[:, 1:4, :] = get_T(center)
    rb[:, 4:7, :] = get_R(axis, angle)    
    
    index = rb_d.shape[0]
    rb_map[name] = index
    rb_d = rb if index==0 else cp.append(rb_d, rb, axis=0)

    return rb_d, rb_map

def transform_rb(rb_d, index, center=None, axis=None, angle=None):
    if center:
        rb_d[index, 1:4, :] = get_T(center)
    if axis and angle:
        rb_d[index, 4:7, :]  = get_R(axis, angle)
        
def set_vel_rb(rb_d, index, vel):
    rb_d[index,-1,:2] = vel