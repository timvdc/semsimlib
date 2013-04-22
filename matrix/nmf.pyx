import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def reconstruct(np.ndarray[ITYPE_t, ndim=1] row, np.ndarray[ITYPE_t, ndim=1] col, np.ndarray[DTYPE_t, ndim=1] data, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H):
    cdef int lendata = data.shape[0]
    cdef np.ndarray reconstruct = np.zeros(lendata, dtype=DTYPE)

    for i in range(lendata):
        reconstruct[i] = np.dot(W[row[i],:], H[:,col[i]])
    return reconstruct
    
