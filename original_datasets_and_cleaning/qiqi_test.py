"""
@Author  : Yuqi Liang 梁彧祺
@File    : qiqi_test.py
@Time    : 23/03/2025 15:20
@Desc    :

[⚙] Benchmarking WARD linkage...

[🐍 fastcluster] Time taken: 0.0005s
[⚡ Rust]        Time taken: 0.0560s
[🔍 Comparison] Max diff between matrices: 72.000000
[🧩 Max diff @ row 0, col 0]
  🔹 fastcluster: [119. 131.   0.   2.]
  🔸 RUST       : [47. 95.  0.  2.]
  🔍 Difference: [72. 36.  0.  0.]

[🔢 First 5 rows of fastcluster linkage matrix]:
[[119.         131.           0.           2.        ]
 [ 47.          95.           0.           2.        ]
 [ 12.          24.           1.96080657   2.        ]
 [ 45.         152.           1.96080657   2.        ]
 [ 68.         123.           1.96080657   2.        ]]

[🔢 First 5 rows of RUST linkage matrix]:
[[ 47.          95.           0.           2.        ]
 [119.         131.           0.           2.        ]
 [ 12.          24.           1.96080657   2.        ]
 [ 45.         152.           1.96080657   2.        ]
 [ 68.         123.           1.96080657   2.        ]]

[📈 Cophenetic correlation]
  🐍 fastcluster: 0.752178
  ⚡ Rust       : 0.752178

🧠 Speedup: 0.01x faster than fastcluster

从这个输出来看，现在的 Rust 和 fastcluster 的聚类结果在结构上非常一致，只是合并顺序不同。因此，不做进一步修改。
"""
import time
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet
from fastcluster import linkage as fc_linkage
from sequenzo.clustering.utils.patched_linkage import smart_linkage


def benchmark_linkage(matrix, method="ward"):
    print(f"\n[⚙] Benchmarking {method.upper()} linkage...\n")

    # === FASTCLUSTER ===
    t0 = time.time()
    Z_fast = fc_linkage(squareform(matrix), method=method)
    t1 = time.time()
    fast_time = t1 - t0
    print(f"[🐍 fastcluster] Time taken: {fast_time:.4f}s")

    # === SMART LINKAGE (Rust OR fastcluster) ===
    t0 = time.time()
    Z_alt, backend_name = smart_linkage(squareform(matrix), method=method)
    t1 = time.time()
    alt_time = t1 - t0
    print(f"[⚡ {backend_name}] Time taken: {alt_time:.4f}s")

    # === Compare matrices ===
    if Z_fast.shape == Z_alt.shape:
        diff = np.abs(Z_fast - Z_alt)
        max_diff = np.max(diff)
        print(f"[🔍 Comparison] Max diff between matrices: {max_diff:.6f}")

        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        row_idx, col_idx = max_diff_idx
        print(f"[🧩 Max diff @ row {row_idx}, col {col_idx}]")
        print(f"  🔹 fastcluster: {Z_fast[row_idx]}")
        print(f"  🔸 {backend_name}: {Z_alt[row_idx]}")
        print(f"  🔍 Difference: {diff[row_idx]}")

        print("\n[🔢 First 5 rows of fastcluster linkage matrix]:")
        print(Z_fast[:5])

        print(f"\n[🔢 First 5 rows of {backend_name} linkage matrix]:")
        print(Z_alt[:5])

        from scipy.spatial.distance import pdist
        c_fast, _ = cophenet(Z_fast, pdist(matrix))
        c_alt, _ = cophenet(Z_alt, pdist(matrix))
        print(f"\n[📈 Cophenetic correlation]\n  🐍 fastcluster: {c_fast:.6f}\n  ⚡ {backend_name}: {c_alt:.6f}")

    else:
        print("[!] Linkage matrix shapes differ.")

    print(f"\n🧠 Speedup: {fast_time / alt_time:.2f}x faster than {backend_name}" if alt_time > 0 else "∞")

    return {
        "fastcluster_time": fast_time,
        "alt_time": alt_time,
        "backend": backend_name,
        "speedup": fast_time / alt_time if alt_time > 0 else float("inf")
    }


if __name__ == "__main__":
    # Import necessary libraries
    # Your calling code (e.g., in a script or notebook)

    from sequenzo import *  # Import the package, give it a short alias
    import pandas as pd  # Data manipulation

    # List all the available datasets in Sequenzo
    # Now access functions using the alias:
    print('Available datasets in Sequenzo: ', list_datasets())

    # Load the data that we would like to explore in this tutorial
    # `df` is the short for `dataframe`, which is a common variable name for a dataset
    df = load_dataset('country_co2_emissions')

    # Create a SequenceData object

    # Define the time-span variable
    time_list = list(df.columns)[1:]

    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    # TODO: write a try and error: if no such a parameter, then ask to pass the right ones
    # sequence_data = SequenceData(df, time=time, time_type="year", id_col="country", ids=df['country'].values, states=states)

    sequence_data = SequenceData(df, time=time_list, time_type="year", id_col="country", states=states)

    # You can also replace "OMspell" with "OM/DHD/HAM" and "TRATE" with "CONSTANT"
    om = get_distance_matrix(seqdata=sequence_data,
                             method='OM',
                             sm="TRATE",
                             indel="auto")

    # Replace with your actual OM matrix generation
    from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
    from sequenzo.define_sequence_data import SequenceData

    stats = benchmark_linkage(om.values, method="ward")
    print("\n🧠 Speedup: {:.2f}x faster than fastcluster".format(stats["speedup"]))
