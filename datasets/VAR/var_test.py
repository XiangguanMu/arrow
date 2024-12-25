import pandas as pd
import numpy as np
from bitarray import bitarray
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from utils.synthetic import simulate_var, compare_graphs
import matplotlib.pyplot as plt
import pickle
from itertools import permutations as ipmt
import time

# (n_ts, n_node)
n_nodes = 10
n_ts = 1000
patch_size = 4
n_patches = n_ts-patch_size+1
# lag = 1
trends = {i:bitarray() for i in range(n_nodes)}

for lag in [2]:
# for lag in [3,4,5,6,7]:
    for _ in range(10):
    # for _ in range(10):
        seed = int(time.time()) % 10000
        data, beta, GC = simulate_var(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        perf = []
        from benchmarks.pcmci import pcmci
        graph = pcmci(data)
        # print("True graph:")
        # print(GC)
        # print("Estimated graph:")
        # print(graph)
        perf.append(compare_graphs(GC, graph))  # tpr, fdr
    print("Means and standard deviations for TPR, FDR and AUC with", lag, "time interval")
    print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))
        