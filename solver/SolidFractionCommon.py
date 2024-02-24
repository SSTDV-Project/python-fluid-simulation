import cupy as cp
from numba import cuda

@cuda.jit(device=True)
def edge_in_fraction(lval, rval):
    l_in = lval < 0
    r_in = rval < 0
    if l_in and r_in:
        return 1
    if not l_in and not r_in:
        return 0
    diff = -abs(lval - rval)
    if l_in and not r_in:
        return lval / diff
    # if not l_in and r_in:
    return rval / diff

@cuda.jit(device=True)
def tri_in_fraction(v0, v1, v2):
    v = cuda.local.array(3, dtype=cp.float64)
    v[0] = v0
    v[1] = v1
    v[2] = v2
    v0_in = v0 < 0
    v1_in = v1 < 0
    v2_in = v2 < 0
    in_count = int(v0_in) + int(v1_in) + int(v2_in)
    
    if in_count == 3:
        return 1.0
    if in_count == 2:
        out_v = 0
        if v0_in: 
            out_v = 1
            if v1_in:
                out_v = 2
        k1 = (out_v + 1) % 3
        k2 = (out_v + 2) % 3
        return 1.0 - edge_in_fraction(v[k1], v[k2])
    if in_count == 1:
        in_v = 0
        if not v0_in:
            in_v = 1
            if not v1_in:
                in_v = 2
        k1 = (in_v + 1) % 3
        k2 = (in_v + 2) % 3
        return edge_in_fraction(v[k1], v[k2])
    if in_count == 0:
        return 0.0
    
@cuda.jit(device=True)
def face_in_fraction(bl, br, tl, tr):
    ce = 0.25 * (bl + br + tl + tr)
    return 0.25 * (
        tri_in_fraction(bl, br, ce)
        + tri_in_fraction(br, tr, ce)
        + tri_in_fraction(tr, tl, ce)
        + tri_in_fraction(tl, bl, ce)
    )
