import numpy as np
cimport numpy as cnp
from sequenzo.define_sequence_data import SequenceData

def seqdur(seqdata):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("data is not a sequence object, see SequenceData to create one")

    cdef cnp.ndarray[long, ndim=2] seqdatanum = seqdata.values.copy()
    seqdatanum[np.isnan(seqdatanum)] = -99

    cdef int n = seqdatanum.shape[0], m = seqdatanum.shape[1]
    cdef int i, j

    cdef cnp.ndarray[long, ndim=2] ffill = seqdatanum.copy()
    for j in range(1, m):
        for i in range(n):
            if seqdatanum[i, j] == -99:
                ffill[i, j] = ffill[i, j - 1]

    cdef cnp.ndarray[char, ndim=2] boundaries = np.ones((n, 1), dtype=bool)
    boundaries = np.concatenate([boundaries, ffill[:, 1:] != ffill[:, :-1]], axis=1)

    cdef cnp.ndarray[long, ndim=2] group_ids = np.cumsum(boundaries, axis=1)
    cdef cnp.ndarray[char, ndim=2] valid = seqdatanum != -99

    cdef list group_durations = []
    cdef int max_groups = 0

    for i in range(n):
        counts = np.bincount(group_ids[i][valid[i]])
        counts = counts[1:] if counts.size > 0 else np.array([], dtype=int)
        group_durations.append(counts)
        max_groups = max(max_groups, len(counts))

    cdef cnp.ndarray[double, ndim=2] result = np.full((n, max_groups), np.nan)

    for i in range(n):
        counts = group_durations[i]
        result[i, :len(counts)] = counts

    return result