"""
@Author  : 李欣怡
@File    : seqdur.py
@Time    : 2024/11/12 00:20
@Desc    : Extracts states durations from sequences
"""
import numpy as np
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength
from sequenzo.define_sequence_data import SequenceData

# example:
#     input:
#         A-A-A-B-B-C-D
#         A-B-B-B-B-B-B
#         B-B-B-B-C-C-C
#     output:
#         3-2-1-1
#         1-6
#         4-3
def seqdur(seqdata):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("data is not a sequence object, see SequenceData to create one")

    seqdatanum = seqdata.values.copy()
    seqdatanum[np.isnan(seqdatanum)] = -99

    n, m = seqdatanum.shape

    ffill = seqdatanum.copy()
    for j in range(1, m):
        ffill[:, j] = np.where(seqdatanum[:, j] != -99, seqdatanum[:, j], ffill[:, j - 1])

    boundaries = np.concatenate([np.ones((n, 1), dtype=bool), ffill[:, 1:] != ffill[:, :-1]], axis=1)
    group_ids = np.cumsum(boundaries, axis=1)

    valid = seqdatanum != -99

    group_durations = []
    max_groups = 0
    for i in range(n):
        counts = np.bincount(group_ids[i][valid[i]])

        if counts.size > 0:
            counts = counts[1:]
        else:
            counts = np.array([], dtype=int)
        group_durations.append(counts)

        if len(counts) > max_groups:
            max_groups = len(counts)

    result = np.full((n, max_groups), np.nan, dtype=float)
    for i, counts in enumerate(group_durations):
        result[i, :len(counts)] = counts

    return result
