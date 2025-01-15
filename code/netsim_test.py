import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import torch
import numpy as np
import csv
from patch import estimate_lags
from tqdm import *
import time
import argparse
from benchmarks.pcmci import pcmci
from benchmarks.surd import surd
from benchmarks.cmlp import ngc
from benchmarks.varlingam import varlingam
from utils.synthetic import compare_graphs

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
parser.add_argument("--data", type=str, default='patched', choices=['raw', 'patched'], help="data mode: use raw data or patched data")
parser.add_argument("--n", type=int, default=10, help="number of nodes")
parser.add_argument("--model", type=str, default='pcmci', choices=['pcmci', 'surd', 'ngc', 'varlingam'], help="{pcmci, surd, ngc, varlingam}")
# args = parser.parse_args(args=[])  # debug mode
args = parser.parse_args()


perf = []
execute_times = []
for i in trange(50):
    fileName = "../datasets/netsim/sim3_subject_%s.npz" % (i)
    ld = np.load(fileName)
    data, GC = ld['X_np'], ld['Gref']
    data = data.T
    n_ts, n_nodes = data.shape
    patch_size = 3
    lag_max = int(0.1*n_ts)

    time_start = time.perf_counter()
    if args.data == 'patched':
        data_bit, nlags, top_indices = estimate_lags(data, patch_size, None, lag_max)
        estimated_lag = np.zeros_like(nlags)
        estimated_lag[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
        if args.model == 'pcmci':
            graph = pcmci(data_bit, nlags, top_indices, use_raw=False)
        if args.model == 'surd':
            graph = surd(data_bit, nlags, top_indices,use_raw=False, use_cp=use_cp)
        if args.model == 'ngc':
            graph, epoch_times = ngc(data, nlags, top_indices, use_raw=False, use_cp=use_cp)
            # graph, epoch_times = ngc(data_bit, nlags, top_indices,use_raw=False, use_constant=args.lag=='constant', use_cp=use_cp)
            execute_times.extend(epoch_times)
            perf.append(compare_graphs(GC, graph))
            continue
        if args.model == 'varlingam':
            graph = varlingam(data_bit, nlags, top_indices, use_raw=False)
        time_end = time.perf_counter()
        execute_times.append(time_end-time_start)
        perf.append(compare_graphs(GC, graph))
    if args.data == 'raw':
        if args.model == 'pcmci':
            graph, estimated_lag = pcmci(data, use_raw=True)
        if args.model == 'surd':
            graph, estimated_lag = surd(data, use_raw=True, use_cp=use_cp)
        if args.model == 'ngc':
            graph, estimated_lag, epoch_times = ngc(data, use_raw=True, use_cp=use_cp)
            execute_times.extend(epoch_times)
            perf.append(compare_graphs(GC, graph))
        if args.model == 'varlingam':
            graph, estimated_lag = varlingam(data, use_raw=True)
        time_end = time.perf_counter()
        perf.append(compare_graphs(GC, graph))
        execute_times.append(time_end-time_start)
        continue

print("Means and standard deviations for TPR, FPR and AUC")
print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))

print("Means and standard deviations for execution time(s)")
print(np.mean(execute_times), np.std(execute_times))