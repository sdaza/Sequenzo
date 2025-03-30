import numpy as np
cimport numpy as cnp
from sequenzo.define_sequence_data import SequenceData

def seqdss(seqdata, bint with_missing=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData to create one.")

    cdef cnp.ndarray[long, ndim=2] seqdatanum = seqdata.values.copy()
    cdef int n = seqdatanum.shape[0], m = seqdatanum.shape[1]
    cdef int i, j

    if not with_missing:
        seqdatanum[np.isnan(seqdatanum)] = -99

    cdef cnp.ndarray[long, ndim=2] ffill = seqdatanum
    cdef cnp.ndarray[char, ndim=2] boundaries = np.ones((n, 1), dtype=bool)

    for j in range(1, m):
        mask = (ffill[:, j] == -99)
        ffill[mask, j] = ffill[mask, j - 1]

    boundaries = np.concatenate([boundaries, ffill[:, 1:] != ffill[:, :-1]], axis=1)

    cdef list groups = [row[boundary & (row != -99)] for row, boundary in zip(ffill, boundaries)]
    cdef int max_groups = max(len(g) for g in groups)

    cdef cnp.ndarray[long, ndim=2] result = np.full((n, max_groups), np.nan, dtype=np.int64)

    for i in range(n):
        g = groups[i]
        result[i, :len(g)] = g

    return result
