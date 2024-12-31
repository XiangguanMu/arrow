import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
from patch import estimate_lags
from tqdm import *
import time
import argparse
from utils.synthetic import simulate_er, compare_graphs, compare_graphs_lag
from benchmarks.pcmci import pcmci_raw,pcmci


parser = argparse.ArgumentParser()
parser.add_argument("--lag", type=str, default='constant', help="lag mode: constant lag or multiple lags")
parser.add_argument("--data", type=str, default='raw', help="data mode: use raw data or patched data")
args = parser.parse_args(args=[])
# (n_ts, n_node)
n_nodes = 10
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)  # to adjust

print(f'====================Lag mode: {args.lag}, Data mode: {args.data}====================')

for lag_range in [1,3,5,7,9,15,20]:
    # lag[i][j] is the lag from j to i
    if args.lag == 'constant':
        lag = np.ones((n_nodes,n_nodes), dtype=int)*lag_range
    elif args.lag == 'multiple':
        lag = np.random.randint(0, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    # print(lag)
    perf = []
    lag_perf = []
    execute_times = []
    for i in trange(10):
        seed = np.random.SeedSequence().generate_state(1)[0]
        data, beta, GC = simulate_er(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        GC_lag = GC*lag
        data_bit, nlags, top_indices = estimate_lags(data, patch_size, lag, lag_max)
        time_start = time.perf_counter()
        # patched
        if args.data == 'patched':
            graph = pcmci(data_bit,nlags=nlags, top_indices=top_indices,use_constant=args.lag)
            time_end = time.perf_counter()
            execute_times.append(time_end-time_start)
            perf.append(compare_graphs(GC, graph))
            top_lags = np.zeros_like(nlags)
            top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
            lag_perf.append(compare_graphs_lag(GC, GC_lag, top_lags))
        # raw
        elif args.data == 'raw':
            graph,estimated_lag = pcmci(data, use_raw=True,use_constant=args.lag)
            time_end = time.perf_counter()
            perf.append(compare_graphs(GC, graph))
            execute_times.append(time_end-time_start)
            if estimated_lag is not None:
                lag_perf.append(compare_graphs_lag(GC, GC_lag, estimated_lag))
    print("Means and standard deviations for TPR, FDR and AUC with lag range in ", lag_range, "time interval")
    print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))
    if len(lag_perf)>0:
        print("Means and standard deviations for lag accuracy with lag range in ", lag_range, "time interval")
        print(np.mean(lag_perf), np.std(lag_perf))
    print("Means and standard deviations for execution time with lag range in ", lag_range, "time interval")
    print(np.mean(execute_times), np.std(execute_times))
