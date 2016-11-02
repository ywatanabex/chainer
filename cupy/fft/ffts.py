import cupy
from cupy.cuda import cufft

CUFFT_C2C = 0x29
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1


CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29


def fft(a):
    a = a.astype('complex64')
    out = cupy.empty_like(a)
    n, d = a.shape
    #plan = cufft.plan1d(d, CUFFT_C2C, n)
    plan = cufft.create()
    cufft.makePlan1d(plan, d, CUFFT_C2C, n)
    cufft.execC2C(plan, a.data.ptr, out.data.ptr, CUFFT_FORWARD)
    cufft.destroy(plan)
    return out


def ifft(a):
    a = a.astype('complex64')
    out = cupy.empty_like(a)
    n, d = a.shape
    #plan = cufft.plan1d(d, CUFFT_C2C, n)
    plan = cufft.create()
    cufft.makePlan1d(plan, d, CUFFT_C2C, n)
    cufft.execC2C(plan, a.data.ptr, out.data.ptr, CUFFT_INVERSE)
    cufft.destroy(plan)
    out /= d
    return out

