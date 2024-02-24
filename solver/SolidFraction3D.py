import cupy as cp
from numba import cuda

from .SolidFractionCommon import *

@cuda.jit
def compute_solid_frac_kernel(gres, sphi, wx, wy, wz):
    x, y, z = cuda.grid(3)
    if x > gres[0]-1 or y > gres[1]-1 or z > gres[2]-1:
        return

    sphi_blb = sphi[2*x  , 2*y  , 2*z  ]
    sphi_brb = sphi[2*x+2, 2*y  , 2*z  ]
    sphi_tlb = sphi[2*x  , 2*y+2, 2*z  ]
    sphi_trb = sphi[2*x+2, 2*y+2, 2*z  ]
    sphi_blf = sphi[2*x  , 2*y  , 2*z+2]
    sphi_brf = sphi[2*x+2, 2*y  , 2*z+2]
    sphi_tlf = sphi[2*x  , 2*y+2, 2*z+2]
    # sphi_trf = sphi[2*x+2, 2*y+2, 2*z+2]
    
    # wx[x+1,y,z] = 1.0 - face_in_fraction(sphi_trb, sphi_brb, sphi_trf, sphi_brf)
    wx[x,y,z] = 1.0 - face_in_fraction(sphi_tlb, sphi_blb, sphi_tlf, sphi_blf)
    # wy[x,y+1,z] = 1.0 - face_in_fraction(sphi_trb, sphi_tlb, sphi_trf, sphi_tlf)
    wy[x,y,z] = 1.0 - face_in_fraction(sphi_brb, sphi_blb, sphi_brf, sphi_blf)
    # wz[x,y,z+1] = 1.0 - face_in_fraction(sphi_trf, sphi_tlf, sphi_brf, sphi_blf)
    wz[x,y,z] = 1.0 - face_in_fraction(sphi_trb, sphi_tlb, sphi_brb, sphi_blb)

def compute_solid_frac(gres, sphi, wx, wy, wz):
    THREAD3 = (8,8,8)
    threads = cp.array(THREAD3, dtype=cp.int64)
    blocks = (*((gres - 1) // threads + 1).get(),)
    compute_solid_frac_kernel[blocks, THREAD3](gres, sphi, wx, wy, wz)