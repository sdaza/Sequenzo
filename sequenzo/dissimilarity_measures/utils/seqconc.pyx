from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cdef str sconc_np(cnp.ndarray[long, ndim=1] seqdata, str sep):
    cdef int i, size = seqdata.shape[0]
    cdef list valid_values = []

    for i in range(size):
        if not np.isnan(seqdata[i]):
            valid_values.append(str(seqdata[i]))

    return sep.join(valid_values)

def seqconc(cnp.ndarray[long, ndim=2] data, str sep="-"):
    if data.ndim == 1:
        return sconc_np(data, sep)
    elif data.ndim == 2:
        return np.array([sconc_np(row, sep) for row in data])
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")