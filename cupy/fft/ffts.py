import cupy
import cufft

CUFFT_C2C = 0x29
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1


CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29


def fft(a):
    out = cupy.empty_like(a)
    n, d = a.shape
    plan = cufft.plan1d(d, CUFFT_C2C, n)
    cufft.execC2C(plan, a.data.ptr, out.data.ptr, CUFFT_FORWARD)
    return out


def fft(a):
    out = cupy.empty_like(a)
    n, d = a.shape
    plan = cufft.plan1d(d, CUFFT_C2C, n)
    cufft.execC2C(plan, a.data.ptr, out.data.ptr, CUFFT_INVERSE)
    return out

