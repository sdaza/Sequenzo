"""
@Author  : Yuqi Liang 梁彧祺
@File    : qiqi_large_data_hclut_test.py
@Time    : 23/03/2025 16:06
@Desc    : 
"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet
from fastcluster import linkage as fc_linkage
from sequenzo.clustering.utils.patched_linkage import linkage as patched_linkage
from sequenzo import * # Import the package, give it a short alias
import pandas as pd


def benchmark_linkage(matrix, method="ward"):
    print(f"\n[⚙] Benchmarking {method.upper()} linkage...\n")

    # === FASTCLUSTER ===
    t0 = time.time()
    Z_fast = fc_linkage(squareform(matrix), method=method)
    t1 = time.time()
    fast_time = t1 - t0
    print(f"[🐍 fastcluster] Time taken: {fast_time:.4f}s")

    # === RUST ===
    t0 = time.time()
    Z_rust = patched_linkage(squareform(matrix), method=method)
    t1 = time.time()
    rust_time = t1 - t0
    print(f"[⚡ Rust]        Time taken: {rust_time:.4f}s")

    # === Compare matrices ===
    if Z_fast.shape == Z_rust.shape:
        diff = np.abs(Z_fast - Z_rust)
        max_diff = np.max(diff)
        print(f"[🔍 Comparison] Max diff between matrices: {max_diff:.6f}")

        # 结构一致性验证（cophenetic correlation）
        from scipy.spatial.distance import pdist
        c_fast, _ = cophenet(Z_fast, pdist(matrix))
        c_rust, _ = cophenet(Z_rust, pdist(matrix))
        print(f"\n[📈 Cophenetic correlation]\n  🐍 fastcluster: {c_fast:.6f}\n  ⚡ Rust       : {c_rust:.6f}")

    else:
        print("[!] Linkage matrix shapes differ.")

    return {
        "fastcluster_time": fast_time,
        "rust_time": rust_time,
        "speedup": fast_time / rust_time if rust_time > 0 else float("inf")
    }


if __name__ == "__main__":
    # CSV 文件路径列表
    csv_files = [
        'df_sampled_500_detailed_sequences.csv',
        'df_sampled_1000_detailed_sequences.csv',
        'df_sampled_2000_detailed_sequences.csv',
        'df_sampled_3000_detailed_sequences.csv',
        'df_sampled_4000_detailed_sequences.csv',
        'df_sampled_5000_detailed_sequences.csv',
        'df_sampled_10000_detailed_sequences.csv',
        'df_sampled_15000_detailed_sequences.csv',
        'df_sampled_25000_detailed_sequences.csv'
    ]

    data_dir = '/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_results/relative_frequency/240210_relative_frequency_test_results'

    # 存储运行时间和文件信息的列表
    results = []

    # 循环读取每个 CSV 文件并计算运行时间
    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)

        _time = list(df.columns)[4:]
        states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware',
                  'support & test']
        df = df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']]

        data = SequenceData(df, time=_time, time_type="age", id_col="worker_id", states=states)
        om = get_distance_matrix(seqdata=data, method="OMspell", sm="TRATE", indel="auto")

        stats = benchmark_linkage(om.values, method="ward")
        print("\n🧠 Speedup: {:.2f}x faster than fastcluster".format(stats["speedup"]))

        # 提取数据量信息
        num_sequences = len(df)

        results.append({
            "filename": filename,
            "num_sequences": num_sequences,
            "fastcluster_time": stats["fastcluster_time"],
            "rust_time": stats["rust_time"],
            "speedup": stats["speedup"]
        })

    # 绘制折线图
    plt.figure(figsize=(12, 6))

    # 绘制 fastcluster 和 rust 的运行时间
    sns.lineplot(x=[r["num_sequences"] for r in results], y=[r["fastcluster_time"] for r in results], marker='o', label="fastcluster")
    sns.lineplot(x=[r["num_sequences"] for r in results], y=[r["rust_time"] for r in results], marker='o', label="rust")

    # 绘制加速比
    ax2 = plt.twinx()
    sns.lineplot(x=[r["num_sequences"] for r in results], y=[r["speedup"] for r in results], marker='x', color='g', label="speedup", ax=ax2)

    plt.xlabel("Number of Sequences")
    plt.ylabel("Runtime (seconds)")
    ax2.set_ylabel("Speedup (rust/fastcluster)")
    plt.title("Runtime and Speedup vs Number of Sequences")
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()