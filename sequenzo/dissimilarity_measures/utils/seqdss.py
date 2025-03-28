"""
@Author  : Xinyi Li 李欣怡
@File    : seqdss.py
@Time    : 2024/11/19 10:11
@Desc    : Extracts distinct states from sequences
"""

import numpy as np
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength


def seqdss(seqdata, with_missing=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData to create one.")

    seqdatanum = seqdata.values.copy()

    if not with_missing:
        seqdatanum[np.isnan(seqdatanum)] = -99

    n, m = seqdatanum.shape

    # process NaN values
    ffill = seqdatanum
    for j in range(1, m):
        mask = (ffill[:, j] == -99)
        ffill[mask, j] = ffill[mask, j - 1]

    boundaries = np.concatenate([np.ones((n, 1), dtype=bool), ffill[:, 1:] != ffill[:, :-1]], axis=1)

    groups = [row[boundary & (row != -99)] for row, boundary in zip(ffill, boundaries)]

    max_groups = max(len(g) for g in groups)

    result = np.full((n, max_groups), np.nan)
    for i, g in enumerate(groups):
        result[i, :len(g)] = g

    return result
