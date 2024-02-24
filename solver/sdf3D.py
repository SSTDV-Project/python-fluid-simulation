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
    for i in range(3):
        for j in range(3):
            TR[i,j]=R[i,j]
        TR[i,3] = T[i,3]

@cuda.jit
def matvecmul4(A, B, C):
    """Perform square matrix multiplication of C = A * B in 4by4 matrix 3 vec C: 3 vec"""  
        
    for i in range(3):
        tmp = 0.
        for j in range(3):
            tmp += A[i,j]*B[j]
        tmp += A[i,3]
        C[i]=tmp

@cuda.jit
def inv_rigid(T, R, inv_tr):
    """inv_rigid in 4*4 size"""  
    for i in range(3):
        tmp = 0.
        for j in range(3):
            inv_tr[i,j]=R[j,i]
            tmp -= R[j,i]*T[j, 3]
        inv_tr[i, 3] = tmp
    inv_tr[3,3]=1
        

@cuda.jit
def norm(pos):
    return (pos[0]**2+pos[1]**2+pos[2]**2)**0.5

@cuda.jit
def vec3_norm(pos, norm_pos):
    norm = (pos[0]**2+pos[1]**2+pos[2]**2)**0.5
    norm_pos[0]=pos[0]/norm
    norm_pos[1]=pos[1]/norm
    norm_pos[2]=pos[2]/norm
    
@cuda.jit
def sphere_eval(rb, position):
    disp = cuda.local.array(3, dtype=cp.float64)
    
    disp[0] = position[0] - rb[1,3]
    disp[1] = position[1] - rb[2,3]
    disp[2] = position[2] - rb[3,3]
    dist = norm(disp)
    disp_normalized = cuda.local.array(3, dtype=cp.float64)
    vec3_norm(disp,disp_normalized)
    sd = dist - rb[0, 1] # rb[0,1]: radius
    if rb[0, 0]%2: #flipped
        sd = -sd
    return sd
    
@cuda.jit
def sphere_project(rb, position):
    #sd, disp_normalized = sphere_eval(rb, position)
    disp = cuda.local.array(3, dtype=cp.float64)
    disp[0] = position[0] - rb[1,3]
    disp[1] = position[1] - rb[2,3]
    disp[2] = position[2] - rb[3,3]
    dist = norm(disp)
    disp_normalized = cuda.local.array(3, dtype=cp.float64)
    vec3_norm(disp,disp_normalized)
    sd = dist - rb[0, 1] # rb[0,1]: radius
    if rb[0, 0]%2: #flipped
        sd = -sd
    if sd < 0:
        for i in range(3):
            position[i] = disp_normalized[i] * rb[0,1] + rb[i+1,3]
        #position = disp_normalized * rb['rbparam'][1] + rb['T'][:3,3]

@cuda.jit
def box_eval(rb, position):
    
    inv_TR = cuda.local.array((4,4), dtype=cp.float64)
    inv_rigid(rb[1:5,:], rb[5:9,:], inv_TR)
    pos_rb = cuda.local.array(3, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    
    disp = cuda.local.array(3, dtype=cp.float64)
    
    for i in range(3):
        disp[i] = abs(pos_rb[i]) - rb[0, 1+i]/2
    
    tmp=0.0
    max_disp=-100
    for i in range(3):
        if disp[i]>0: tmp += disp[i]**2
        if max_disp < disp[i]: max_disp = disp[i]
    sd = tmp ** 0.5 
    if max_disp < 0: sd += max_disp
    
    if rb[0, 0] % 2: #flipped
        sd = -sd
    return sd
    
@cuda.jit
def box_project(rb, position):
    inv_TR = cuda.local.array((4,4), dtype=cp.float64)
    inv_rigid(rb[1:5,:], rb[5:9,:],inv_TR)
    pos_rb = cuda.local.array(3, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    
    in_out = 0 # 0 means in 
    for i in range(3):
        if pos_rb[i]> rb[0, 1+i]/2 or pos_rb[i]< -rb[0, 1+i]/2:
            in_out +=1

    if rb[0, 0] % 2 and ~(in_out): # flipped, and in
        for i in range(3): #pos_rb[:3] = cp.clip(pos_rb[:3],-half_size, half_size)
            if pos_rb[i]<-rb[0, 1+i]/2: pos_rb[i] = -rb[0, 1+i]/2
            elif pos_rb[i]>rb[0, 1+i]/2: pos_rb[i] = rb[0, 1+i]/2
        TR = cuda.local.array((4,4), dtype=cp.float64)
        mat_TR(rb[1:5,:], rb[5:9,:], TR)
        matvecmul4(TR, pos_rb, position)

    elif in_out==0: # not flipped, and in
        index=0 # 0:-x, 1:+x, 2:-y, 3:+y, 4:-z, 5:+z
        dist_xyz=100
        for i in range(3):
            if rb[0,1+i]/2 - pos_rb[i] < dist_xyz:
                dist_xyz = rb[0,1+i]/2 - pos_rb[i]
                index = i*2
            if pos_rb[i]+ rb[0,1+i]/2 < dist_xyz:
                dist_xyz = pos_rb[i]+ rb[0,1+i]/2
                index = i*2+1
        pos_rb[index//2] += dist_xyz * (-1)**(index%2)
        TR = cuda.local.array((4,4), dtype=cp.float64)
        mat_TR(rb[1:5,:], rb[5:9,:], TR)
        matvecmul4(TR, pos_rb, position)    

@cuda.jit
def cylinder_eval(rb, position):
    inv_TR = cuda.local.array((4,4), dtype=cp.float64)
    inv_rigid(rb[1:5,:], rb[5:9,:],inv_TR)
    pos_rb = cuda.local.array(3, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    half_height = rb[0, 2]/2
    
    if pos_rb[1]<-half_height: y_clip=-half_height
    elif pos_rb[1]>half_height: y_clip=half_height
    
    sd = (pos_rb[0]**2+pos_rb[2]**2)**0.5 - rb[0, 1]
    
    if sd<0:
        if y_clip == half_height or y_clip == -half_height: #above/below cylinder
            sd = abs(y_clip - pos_rb[1])
        else: #inside
            sd = max(sd, pos_rb[1]-half_height, -(pos_rb[1]+half_height))
    else: # outside cylinder
        if y_clip == half_height or y_clip == -half_height: #above/below cylinder
            del_y= abs(y_clip-pos_rb[1])
            sd = (sd**2+del_y**2)**0.5
    
    if rb[0, 0] % 2: #flipped
        sd = -sd
    return sd

@cuda.jit
def cylinder_project(rb, position):
    inv_TR = cuda.local.array((4,4), dtype=cp.float64)
    inv_rigid(rb[1:5,:], rb[5:9,:],inv_TR)
    pos_rb = cuda.local.array(3, dtype=cp.float64)
    matvecmul4(inv_TR, position, pos_rb)
    half_height = rb[0, 2]/2
    
    y_clip = pos_rb[1]   
    if pos_rb[1]<-half_height: y_clip=-half_height
    elif pos_rb[1]>half_height: y_clip=half_height
    
    dist = (pos_rb[0]**2+pos_rb[2]**2)**0.5 
    sd = dist - rb[0,1]

    if rb[0, 0] % 2:
        if abs(y_clip) == half_height or sd>0: #outside
            if sd<0: #above/ below cylinder
                pos_rb[1] = y_clip
            else: # project to side face
                pos_rb[0] = pos_rb[0] / dist * rb [0,1]
                pos_rb[2] = pos_rb[2] / dist * rb [0,1]
                pos_rb[1] = y_clip
                
        TR = cuda.local.array((4,4), dtype=cp.float64)
        mat_TR(rb[1:5,:], rb[5:9,:], TR)
        matvecmul4(TR, pos_rb, position)
        #position = cp.matmul(cp.matmul(rb['T'],rb['R']) , pos_rb)[:3]
    else:
        if sd < 0 and abs(y_clip) != half_height: #inside
            max_value= max(sd, pos_rb[1]-half_height, -(pos_rb[1]+half_height))
            if max_value==sd:
                pos_rb[0] = pos_rb[0] /dist * rb [0,1]
                pos_rb[2] = pos_rb[2] /dist * rb [0,1]
            elif max_value == pos_rb[1]-half_height:
                pos_rb[1] = half_height
            else:
                pos_rb[1] = -half_height
            TR = cuda.local.array((4,4), dtype=cp.float64)
            mat_TR(rb[1:5,:], rb[5:9,:], TR)
            matvecmul4(TR, pos_rb, position)  
            #position = cp.matmul(cp.matmul(rb['T'],rb['R']) , pos_rb)[:3]
    return position


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
        elif rb_d[i,0,0]//2 == 2: #cylinder
            d = cylinder_eval(rb_d[i,:,:], position[P,:])
        if d < min_sd: 
            min_sd = d
            rb_index = i
    sd[P] = min_sd
    if min_sd <= 0:
        for i in range(3):
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
           
        elif rb_d[i, 0, 0]//2 == 2: # cylinder
            cylinder_project(rb_d[i,:,:], position[P,:])

def evaluate(rb_d, sd, vel, position):
    assert sd.shape == position.shape[:-1]
    assert position.shape[-1] == 3
    assert vel.shape[-1] == 3
    
    vel *= 0
    NUM_POSITIONS = position.reshape(-1,3).shape[0]
    
    THREAD1 = 256
    blocks = ((NUM_POSITIONS-1) // THREAD1) + 1
    evaluate_kernel[blocks, THREAD1](rb_d, sd.reshape(-1), vel.reshape(-1,3), position.reshape(-1,3))
    
def project(rb_d, position):
    assert position.shape[-1] == 3
    
    THREAD1 = 256
    blocks = ((position.shape[0]-1) // THREAD1) + 1
    project_kernel[blocks, THREAD1](rb_d, position)

def get_T(position):
    t_mat = cp.identity((4))
    t_mat[0:3,3] = cp.asarray(position)
    return t_mat

def get_R(axis, angle):
    r_mat = cp.identity((4))
    if angle:
        r_mat3 = R.from_rotvec(axis / np.linalg.norm(axis) * angle * np.pi / 180).as_matrix()
        r_mat[:3,:3]=cp.asarray(r_mat3)
    return r_mat
    

def generate_rb(rb_d, rb_map, name, rbparam, flip=False, center=[0, 0, 0], axis=[0, 1, 0], angle=0):
    # arguments
    # rb_d: all rb information
    # rbparam: ['sphere', radius], ['box', x_scale, y_scale, z_scale], ['cylinder', radius, height]
    # rb_d: number of rb * 9 * 4 matrix
    # one rb has size 9 by 4 matrix
    # row 0: rbparam 
        # sphere: [0, radius, 0, 0], 1 means flipped
        # box: [2, x_scale, y_scale, z_scale], 3 means flipped
        # cylinder: [4, radius, height, 0] center at origin, 5 means flipped
    # row 1-4: translation matrix in mat4
    # row 5-8: rotation axis and angle in mat4
    
    rb = cp.zeros((1, 10, 4))
    if rbparam[0]=='sphere':
        rb[:, 0, 0] = 1 if flip else 0
        rb[:, 0, 1] = rbparam[1]
    elif rbparam[0]=='box':
        rb[:, 0, 0] = 3 if flip else 2
        rb[:, 0, 1:] = cp.asarray(rbparam[1:])
    elif rbparam[0]=='cylinder':
        rb[:, 0, 0] = 5 if flip else 4
        rb[:, 0, 1:3] = cp.asarray(rbparam[1:])
    else:
        return rb_d
    
    rb[:, 1:5, :] = get_T(center)
    rb[:, 5:9, :] = get_R(axis, angle)    
    
    index = rb_d.shape[0]
    rb_map[name] = index
    rb_d = rb if index==0 else cp.append(rb_d, rb, axis=0)

    return rb_d, rb_map

def transform_rb(rb_d, index, center=None, axis=None, angle=None):
    if center:
        rb_d[index, 1:5, :] = get_T(center)
    if axis and angle:
        rb_d[index, 5:9, :]  = get_R(axis, angle)
        
def set_vel_rb(rb_d, index, vel):
    rb_d[index,-1,:3] = vel