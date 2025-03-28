"""
@Author  : 李欣怡
@File    : seqconc.py
@Time    : 2024/11/2 18:02
@Desc    : Concatenates vectors of states or events into character strings
"""
import numpy as np
import pandas as pd
from sequenzo.define_sequence_data import SequenceData


def sconc_np(seqdata, sep):
    vi = ~np.isnan(seqdata)  # Choose values that are not NA
    return sep.join(seqdata[vi].astype(str))


def seqconc(data, sep="-"):
    if data.ndim == 1:
        cseq = sconc_np(pd.Series(data), sep)

    elif data.ndim == 2:
        cseq = np.array([sconc_np(row, sep) for row in data])

    else:
        raise ValueError("Only 1D and 2D arrays are supported.")

    return cseq

