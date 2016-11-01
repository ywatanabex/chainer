"""wrapper of cuFFT"""
cimport cython


ctypedef int cufftHandle

###############################################################################
# Extern
###############################################################################

cdef extern from "cufft.h":
    ctypedef struct cufftComplex

    ctypedef enum cufftResult:
        CUFFT_SUCCESS
        CUFFT_INVALID_PLAN
        CUFFT_ALLOC_FAILED
        CUFFT_INVALID_TYPE
        CUFFT_INVALID_VALUE
        CUFFT_INTERNAL_ERROR
        CUFFT_EXEC_FAILED
        CUFFT_SETUP_FAILED
        CUFFT_INVALID_SIZE
        CUFFT_UNALIGNED_DATA
        CUFFT_INCOMPLETE_PARAMETER_LIST
        CUFFT_INVALID_DEVICE
        CUFFT_PARSE_ERROR
        CUFFT_NO_WORKSPACE
        CUFFT_NOT_IMPLEMENTED
        CUFFT_LICENSE_ERROR
        CUFFT_NOT_SUPPORTED
    
    ctypedef enum cufftType:
        CUFFT_R2C
        CUFFT_C2R
        CUFFT_C2C
        CUFFT_D2Z
        CUFFT_Z2D
        CUFFT_Z2Z

    cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
    cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)



###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0x0: 'CUFFT_SUCCESS',
    0x1: 'CUFFT_INVALID_PLAN',
    0x2: 'CUFFT_ALLOC_FAILED',
    0x3: 'CUFFT_INVALID_TYPE',
    0x4: 'CUFFT_INVALID_VALUE',
    0x5: 'CUFFT_INTERNAL_ERROR',
    0x6: 'CUFFT_EXEC_FAILED',
    0x7: 'CUFFT_SETUP_FAILED',
    0x8: 'CUFFT_INVALID_SIZE',
    0x9: 'CUFFT_UNALIGNED_DATA',
    0xA: 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    0xB: 'CUFFT_INVALID_DEVICE',
    0xC: 'CUFFT_PARSE_ERROR',
    0xD: 'CUFFT_NO_WORKSPACE',
    0xE: 'CUFFT_NOT_IMPLEMENTED',
    0x0F: 'CUFFT_LICENSE_ERROR',
    0x10: 'CUFFT_NOT_SUPPORTED',
}


class CUFFTError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super(CUFFTError, self).__init__(STATUS[status])

@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUFFTError(status)
    


###############################################################################
# Extern
###############################################################################



cpdef size_t plan1d(int nx, cufftType type, int batch) except *:
    cdef cufftHandle plan
    cufftPlan1d(&plan, nx, type, batch)
    return <size_t>plan


cpdef execC2C(
    cufftHandle plan, size_t idata, size_t odata, int direction):
    status = cufftExecC2C(plan, <cufftComplex*> idata, <cufftComplex*> odata, direction)
    check_status(status)
