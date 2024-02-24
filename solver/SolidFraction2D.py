import cupy as cp
from numba import cuda

from .SolidFractionCommon import *

@cuda.jit
def compute_solid_frac_kernel(gres, sphi, wx, wy):
    x, y = cuda.grid(2)
    if x >= gres[0]-1 or y >= gres[1]-1:
        return

    sphi_bl = sphi[2*x  , 2*y  ]
    sphi_br = sphi[2*x+2, 2*y  ]
    sphi_tl = sphi[2*x  , 2*y+2]
    sphi_tr = sphi[2*x+2, 2*y+2]
    
    wx[x+1,y] = 1.0 - edge_in_fraction(sphi_tr, sphi_br)
    wx[x,y] = 1.0 - edge_in_fraction(sphi_tl, sphi_bl)
    wy[x,y+1] = 1.0 - edge_in_fraction(sphi_tr, sphi_tl)
    wy[x,y] = 1.0 - edge_in_fraction(sphi_br, sphi_bl)

def compute_solid_frac(gres, sphi, wx, wy):
    THREAD2 = (32, 32)
    threads = cp.array(THREAD2, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    compute_solid_frac_kernel[blocks, THREAD2](gres, sphi, wx, wy)