import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
import cupy as cp
import subprocess
from tqdm import *
import time
import matplotlib.pyplot as plt

import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
from benchmarks.surd import run_surd, surd
import utils.it_tools as it
from itertools import combinations as icmb
from utils.synthetic import simulate_linear, simulate_nonlinear, compare_graphs
from patch import estimate_lags

np.seterr(divide='ignore',invalid='ignore')

def draw_plt(tstname, x_list, y_list):
    plt.figure(figsize=(9, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    draw_size = 32
    draw_width = 3
    ax = plt.gca()
    ax.spines['top'].set_linewidth(draw_width)
    ax.spines['right'].set_linewidth(draw_width)
    ax.spines['bottom'].set_linewidth(draw_width)
    ax.spines['left'].set_linewidth(draw_width)
    plt.yticks(fontproperties = 'Arial', size = draw_size)
    plt.xticks(fontproperties = 'Arial', size = draw_size)

    plt.bar(x_list, y_list)
    plt.show()

    plt.savefig(f'../results/motivations/{tstname}/graph.png')


def pcmci_lag_test(data, t_max):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    results = pcmci.run_pcmci(tau_max=t_max, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
    q_matrix = (q_matrix < 0.05) * 1  # add 00
    return q_matrix.shape


def surd_bins_test(X, nbins):
    n_ts, n_nodes = X.shape
    X = X.T  # n_nodes, n_ts
    graph = np.zeros((n_nodes,n_nodes),dtype=int)
    nlags = np.array([1] * n_nodes)

    for i in range(n_nodes):
        # Organize data (0 target variable, 1: agent variables)
        Y = np.vstack([X[i, nlags[i]:], X[:, :-nlags[i]]])
        # Run SURD
        Y_gpu = cp.asarray(Y)
        hist_gpu, _gpu = cp.histogramdd(Y_gpu.T, bins=nbins)
        hist = cp.asnumpy(hist_gpu)
        I_R, I_S, MI, info_leak = run_surd(hist)
    return len(MI)
                
    

def time_lag_efficiency_test(n_nodes, n_ts, lag_range):
    lag = np.random.randint(1, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    seed = np.random.SeedSequence().generate_state(1)[0]
    data, beta, GC = simulate_nonlinear(p=n_nodes, T=n_ts, lag=lag, seed=seed)
    time_consumed = []
    # lags = range(10, 0, -1)
    lags = range(100, 0, -10)
    print('====begin lag range test====')
    for l in tqdm(lags):
        print(f'lag range: {l}')
        time_start = time.perf_counter()
        _ = pcmci_lag_test(data, t_max=l)
        time_end = time.perf_counter()
        print(f'time cost for lag range {l}(s): {time_end-time_start}')
        time_consumed.append(time_end-time_start)
    tst_dir = '../results/motivations/time_lag_efficiency_test/'
    if not os.path.exists(tst_dir):
        os.makedirs(tst_dir)
    with open(tst_dir+'lists.txt', 'w') as f:
        f.write(f'time consumed: {time_consumed}\n')
        f.write(f'lags: {list(lags)}')
    # draw_plt('time_lag_efficiency_test', x_list=lags, y_list=time_consumed)
    

def bins_cost_test(n_nodes, n_ts, lag_range):
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("GPU is available.")
            use_cp = True
        else:
            msg = "No GPU found or NVIDIA driver is not installed."
            return msg
    except FileNotFoundError:
        use_cp = False
        msg = "nvidia-smi command not found. NVIDIA driver might not be installed."
        return msg

    lag = np.random.randint(1, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    seed = np.random.SeedSequence().generate_state(1)[0]
    data, beta, GC = simulate_nonlinear(p=n_nodes, T=n_ts, lag=lag, seed=seed)
    time_consumed = []

    bins = range(10, 1, -1)  # bin size = 1 is meaningless
    print('====begin bin size test====')
    for b in tqdm(bins):
        print(f'bin range: {b}')
        time_start = time.perf_counter()
        _ = surd_bins_test(data, nbins=b)
        time_end = time.perf_counter()
        time_consumed.append(time_end-time_start)
    tst_dir = '../results/motivations/bins_cost_test/'
    if not os.path.exists(tst_dir):
        os.makedirs(tst_dir)
    with open(tst_dir+'lists.txt', 'w') as f:
        f.write(f'time consumed: {time_consumed}\n')
        f.write(f'bins: {list(bins)}')


def candidate_set_test(n_nodes, n_ts, patch_size, lag_max, lag_range = 3):
    use_cp = False
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("GPU is available.")
            use_cp = True
        else:
            msg = "No GPU found or NVIDIA driver is not installed."
            return msg
    except FileNotFoundError:
        use_cp = False
        msg = "nvidia-smi command not found. NVIDIA driver might not be installed."
        return msg

    pruned_time = []
    pruned_perf = []
    unpruned_time = []
    unpruned_perf = []
    
    for _ in trange(10):
        lag = np.ones((n_nodes,n_nodes), dtype=int)*lag_range
        seed = np.random.SeedSequence().generate_state(1)[0]
        data, beta, GC = simulate_linear(p=n_nodes, T=n_ts, lag=lag, seed=seed)

        data_bit, nlags, top_indices = estimate_lags(data, patch_size, lag, lag_max)
        estimated_lag = np.zeros_like(nlags)
        estimated_lag[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]

        # pruned candidates
        pruned_time_start = time.perf_counter()
        graph_pruned = surd(data_bit, nlags, top_indices,use_raw=False, use_constant=True, use_cp=use_cp, pruned=True)
        pruned_time_end = time.perf_counter()
        pruned_time.append(pruned_time_end-pruned_time_start)
        pruned_perf.append(compare_graphs(GC, graph_pruned))
        
        # unpruned candidates
        unpruned_time_start = time.perf_counter()
        graph_unpruned = surd(data_bit, nlags, top_indices,use_raw=False, use_constant=True, use_cp=use_cp, pruned=False)
        unpruned_time_end = time.perf_counter()
        unpruned_time.append(unpruned_time_end-unpruned_time_start)
        unpruned_perf.append(compare_graphs(GC, graph_unpruned))
    
    pruned_time_mean, pruned_time_std = np.mean(pruned_time), np.std(pruned_time)
    pruned_perf_mean, pruned_perf_std = np.mean(np.reshape(pruned_perf, (-1,3)), axis=0), np.std(np.reshape(pruned_perf, (-1,3)), axis=0)
    unpruned_time_mean, unpruned_time_std = np.mean(unpruned_time), np.std(unpruned_time)
    unpruned_perf_mean, unpruned_perf_std = np.mean(np.reshape(unpruned_perf, (-1,3)), axis=0), np.std(np.reshape(unpruned_perf, (-1,3)), axis=0)
    
    tst_dir = '../results/motivations/candidate_set_test/'
    if not os.path.exists(tst_dir):
        os.makedirs(tst_dir)
    with open(tst_dir+'lists.txt', 'w') as f:
        f.write(f'pruned time (s): {pruned_time_mean, pruned_time_std}\n')
        f.write(f'pruned perf (tpr, fpr, auc): {pruned_perf_mean, pruned_perf_std}\n')
        f.write(f'unpruned time (s): {unpruned_time_mean, unpruned_time_std}\n')
        f.write(f'unpruned perf (tpr, fpr, auc): {unpruned_perf_mean, unpruned_perf_std}\n')

n_nodes = 7  # 7 for bin test
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)
lag_range = 3
bins_cost_test(n_nodes, n_ts, lag_range)
# candidate_set_test(n_nodes, n_ts, patch_size=3, lag_max=lag_max, lag_range=3)
