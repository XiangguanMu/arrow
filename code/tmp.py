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
from benchmarks.pcmci import pcmci
from benchmarks.surd import surd

use_cp = False

import subprocess
try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print("GPU is available.")
        use_cp = True
    else:
        print("No GPU found or NVIDIA driver is not installed.")
except FileNotFoundError:
    use_cp = False
    print("nvidia-smi command not found. NVIDIA driver might not be installed.")


parser = argparse.ArgumentParser()
parser.add_argument("--lag", type=str, default='constant', help="lag mode: constant lag or multiple lags")
parser.add_argument("--data", type=str, default='patched', help="data mode: use raw data or patched data")
parser.add_argument("--n", type=int, default=3, help="number of nodes")
parser.add_argument("--model", type=str, default='surd', help="{pcmci, surd}")
args = parser.parse_args()
# (n_ts, n_node)
n_nodes = args.n
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)  # to adjust

print(f'====================Lag mode: {args.lag}, Data mode: {args.data}, Method: {args.model}====================')

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
        time_start = time.perf_counter()
        graph = None
        # patched
        if args.data == 'patched':
            data_bit, nlags, top_indices = estimate_lags(data, patch_size, lag, lag_max)
            top_lags = np.zeros_like(nlags)
            top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
            if args.model == 'pcmci':
                graph = pcmci(data_bit,nlags=nlags, top_indices=top_indices,use_constant=args.lag)
            if args.model == 'surd':
                graph = surd(data_bit, nlags, top_indices,use_raw=args.data=='raw', use_constant=args.lag=='constant', use_cp=use_cp)
            time_end = time.perf_counter()
            execute_times.append(time_end-time_start)
            perf.append(compare_graphs(GC, graph))
            # print('GC LAG:\n', GC_lag)
            # print('ESTIMATED LAG:\n', top_lags)
            lag_perf.append(compare_graphs_lag(GC, GC_lag, top_lags))
        # raw
        elif args.data == 'raw':
            if args.model == 'pcmci':
                graph, estimated_lag = pcmci(data, use_raw=True,use_constant=args.lag=='constant')
            if args.model == 'surd':
                graph, estimated_lag = surd(data, use_raw=True, use_constant=args.lag=='constant')
                # print('GC LAG:\n', GC_lag)
                # print('ESTIMATED LAG:\n', estimated_lag)
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
