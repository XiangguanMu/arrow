import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
from patch import estimate_lags
from tqdm import *
import time
from utils.synthetic import simulate_er_one_lag, compare_graphs, compare_graphs_lag
from benchmarks.pcmci import pcmci_raw,pcmci



# (n_ts, n_node)
n_nodes = 10
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)  # to adjust

for lag_range in [1,3,5,7,9,15,20]:
    # lag[i][j] is the lag from j to i
    lag = np.random.randint(0, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    # print(lag)
    perf = []
    lag_perf = []
    execute_times = []
    for i in trange(5):
        seed = np.random.SeedSequence().generate_state(1)[0]
        data, beta, GC = simulate_er_one_lag(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        data_bit, nlags, top_indices = estimate_lags(data, patch_size, lag, lag_max)
        time_start = time.perf_counter()
        graph = pcmci(data_bit,nlags=nlags, top_indices=top_indices)
        time_end = time.perf_counter()
        GC_lag = GC*lag
        execute_times.append(time_end-time_start)
        perf.append(compare_graphs(GC, graph))
        top_lags = np.zeros_like(nlags)
        top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
        lag_perf.append(compare_graphs_lag(GC, GC_lag, top_lags))
    print("Means and standard deviations for TPR, FDR and AUC with lag range in ", lag_range, "time interval")
    print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))
    print("Means and standard deviations for lag accuracy with lag range in ", lag_range, "time interval")
    print(np.mean(lag_perf), np.std(lag_perf))
    print("Means and standard deviations for execution time with lag range in ", lag_range, "time interval")
    print(np.mean(execute_times), np.std(execute_times))
