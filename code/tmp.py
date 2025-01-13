import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import torch
import numpy as np
from patch import estimate_lags
from tqdm import *
import time
import argparse
from utils.synthetic import simulate_linear, simulate_nonlinear, compare_graphs, compare_graphs_lag
from benchmarks.pcmci import pcmci
from benchmarks.surd import surd
from benchmarks.cmlp import ngc
from benchmarks.varlingam import varlingam
# import matplotlib.pyplot as plt

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

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--lag", type=str, default='constant', choices=['constant','multiple'], help="lag mode: constant lag or multiple lags")
parser.add_argument("--data", type=str, default='patched', choices=['raw', 'patched'], help="data mode: use raw data or patched data")
parser.add_argument("--dataset", type=str, default='linear', choices=['linear', 'nonlinear'], help="dataset: {linear, nonlinear}")
parser.add_argument("--n", type=int, default=10, help="number of nodes")
parser.add_argument("--model", type=str, default='pcmci', choices=['pcmci', 'surd', 'ngc', 'varlingam'], help="{pcmci, surd, ngc, varlingam}")
# args = parser.parse_args(args=[])  # debug mode
args = parser.parse_args()


def save_results(graph, estimate_lag, GC, GC_lag, i, lag_range):
    path_dir = f'../results/nps/{args.model}_{args.lag}_{args.data}_{n_nodes}_{lag_range}/'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    graph_dir = path_dir+'graph/'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if graph is not None:
        np.save(graph_dir+f'{i}.npy', graph)
    
    estimate_lag_dir = path_dir+'estimate_lag/'
    if not os.path.exists(estimate_lag_dir):
        os.makedirs(estimate_lag_dir)
    if estimate_lag is not None:
        np.save(estimate_lag_dir+f'{i}.npy', estimate_lag)
    
    GC_dir = path_dir+'GC/'
    if not os.path.exists(GC_dir):
        os.makedirs(GC_dir)
    if GC is not None:
        np.save(GC_dir+f'{i}.npy', GC)
    
    GC_lag_dir = path_dir+'GC_lag/'
    if not os.path.exists(GC_lag_dir):
        os.makedirs(GC_lag_dir)
    if GC_lag is not None:
        np.save(GC_lag_dir+f'{i}.npy', GC_lag)
    
    # print('Results saved in ', path_dir, 'round ', i)

# (n_ts, n_node)
n_nodes = args.n
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)  # to adjust

print(f'====================Lag mode: {args.lag}, Data mode: {args.data}, Dataset: {args.dataset}, N:{args.n}, Method: {args.model}====================')

# for lag_range in [5]:
for lag_range in [1,3,5,7,9,15,20]:
    # lag[i][j] is the lag from j to i
    if args.lag == 'constant':
        lag = np.ones((n_nodes,n_nodes), dtype=int)*lag_range
    elif args.lag == 'multiple':
        if lag_range == 1:  # same as constant test
            continue
        lag = np.random.randint(1, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    # print(lag)
    perf = []
    lag_perf = []
    execute_times = []
    for i in trange(10):
        seed = np.random.SeedSequence().generate_state(1)[0]
        if args.dataset == 'nonlinear':
            data, beta, GC = simulate_nonlinear(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        elif args.dataset == 'linear':
            data, beta, GC = simulate_linear(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        # print('====GC====\n', GC)
        GC_lag = GC*lag
        time_start = time.perf_counter()
        graph = None
        # patched
        if args.data == 'patched':
            data_bit, nlags, top_indices = estimate_lags(data, patch_size, lag, lag_max)
            estimated_lag = np.zeros_like(nlags)
            estimated_lag[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
            if args.model == 'pcmci':
                graph = pcmci(data_bit, nlags, top_indices, use_raw=False, use_constant=args.lag=='constant')
            if args.model == 'surd':
                graph = surd(data_bit, nlags, top_indices,use_raw=False, use_constant=args.lag=='constant', use_cp=use_cp)
            if args.model == 'ngc':
                graph, epoch_times = ngc(data, nlags, top_indices, use_raw=False, use_constant=args.lag=='constant', use_linear=args.dataset=='linear', use_cp=use_cp)
                # graph, epoch_times = ngc(data_bit, nlags, top_indices,use_raw=False, use_constant=args.lag=='constant', use_cp=use_cp)
                execute_times.extend(epoch_times)
                perf.append(compare_graphs(GC, graph))
                lag_perf.append(compare_graphs_lag(GC, GC_lag, estimated_lag))
                continue
            if args.model == 'varlingam':
                graph = varlingam(data_bit, nlags, top_indices, use_raw=False, use_constant=args.lag=='constant')

            time_end = time.perf_counter()
            execute_times.append(time_end-time_start)
            perf.append(compare_graphs(GC, graph))
            lag_perf.append(compare_graphs_lag(GC, GC_lag, estimated_lag))
        # raw
        elif args.data == 'raw':
            if args.model == 'pcmci':
                graph, estimated_lag = pcmci(data, use_raw=True,use_constant=args.lag=='constant')
            if args.model == 'surd':
                graph, estimated_lag = surd(data, use_raw=True, use_constant=args.lag=='constant', use_cp=use_cp)
            if args.model == 'ngc':
                graph, estimated_lag, epoch_times = ngc(data, use_raw=True, use_constant=args.lag=='constant', use_linear=args.dataset=='linear', use_cp=use_cp)
                execute_times.extend(epoch_times)
                perf.append(compare_graphs(GC, graph))
                lag_perf.append(compare_graphs_lag(GC, GC_lag, estimated_lag))
                continue     
            if args.model == 'varlingam':
                graph, estimated_lag = varlingam(data, use_raw=True, use_constant=args.lag=='constant')
            time_end = time.perf_counter()
            perf.append(compare_graphs(GC, graph))
            execute_times.append(time_end-time_start)
            if estimated_lag is not None:
                lag_perf.append(compare_graphs_lag(GC, GC_lag, estimated_lag))        
        save_results(graph, estimated_lag, GC, GC_lag, i, lag_range)

    print("Means and standard deviations for TPR, FPR and AUC with lag range in ", lag_range, "time interval")
    print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))

    print("Means and standard deviations for lag TPR, FPR and AUC with lag range in ", lag_range, "time interval")
    print(np.mean(np.reshape(lag_perf, (-1, 3)), axis=0), np.std(np.reshape(lag_perf, (-1, 3)), axis=0))

    print("Means and standard deviations for execution time with lag range in ", lag_range, "time interval")
    print(np.mean(execute_times), np.std(execute_times))
