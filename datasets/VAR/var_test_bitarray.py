import numpy as np
from bitarray import bitarray
from itertools import permutations as ipmt
import time

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from utils.synthetic import simulate_var, simulate_er_one_lag, compare_graphs, compare_graphs_lag
from benchmarks.pcmci import pcmci_raw,pcmci
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# (n_ts, n_node)
n_nodes = 10
n_ts = 1000
patch_size = 3
n_patches = n_ts-patch_size+1
lag_max = int(0.1*n_ts)  # to adjust
groups = 10

# for lag_divide in range(1,groups+1):
for lag_range in [1,3,5,7,9,15,20]:
    # lag[i][j] is the lag from j to i
    lag = np.random.randint(0, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    # print(lag)
# for lag in [1,3,5,7,9,15,20]:
    perf = []
    lag_perf = []
    execute_times = []
    for _ in range(5):
        trends = {i:bitarray() for i in range(n_nodes)}
        seed = np.random.SeedSequence().generate_state(1)[0]
        data, beta, GC = simulate_er_one_lag(p=n_nodes, T=n_ts, lag=lag, seed=seed)
        data_bit = np.zeros((n_patches,n_nodes))
        for i in range(n_nodes):
            _data = data[:,i]
            for j in range(n_patches):
                patch = _data[j:j+patch_size]
                bit0, bit1 = _data[j+1]-_data[j]>0, _data[j+2]-_data[j+1]>0
                trends[i].extend([bit0, bit1])
                data_bit[j][i] = bit0*2+bit1  # 00->0 01->1 10->2 11->3, 4 bins
        max_lags = {pmt:0 for pmt in list(ipmt(range(n_nodes), 2)) if pmt[0]!=pmt[1]}
        signs_0 = np.zeros((lag_max, n_nodes, n_nodes))
        signs_1 = np.zeros((lag_max, n_nodes, n_nodes))
        signs_2 = np.zeros((lag_max, n_nodes, n_nodes))
        for pmt in max_lags.keys():
            i, itrend = pmt[0], trends[pmt[0]]
            j, jtrend = pmt[1], trends[pmt[1]]
            max_lag_ij = 0
            # offset: iterate over [0, n_patches-1]
            # right offset will not be affected by missing leading zero or non-cyclic rolling
            for dt in range(0, n_patches):
                offset = dt*2
                need_len = (n_patches-dt)*2  # 0s in [0:dt*2] stem from right shift; 0s after dt*2 stem from trend encoding
                xor_int = itrend[-need_len:] ^ jtrend[:need_len]  # j->i
                # 11->2 opposite 01->1 non 00->0 same
                correlated_signs = [int(xor_int[i])+int(xor_int[i+1]) for i in range(0, need_len, 2)]
                if dt<lag_max:
                    signs_0[dt][i][j] = correlated_signs.count(0)/len(correlated_signs)
                    signs_1[dt][i][j] = correlated_signs.count(1)/len(correlated_signs)
                    signs_2[dt][i][j] = correlated_signs.count(2)/len(correlated_signs)
        
        signs = np.maximum(signs_0, signs_2)
        
        if isinstance(lag, int):
            nlags = np.zeros(n_nodes, dtype=int)
            ncandidates = np.zeros(n_nodes,dtype=float)
            for i in range(n_nodes):
                candidate = [np.max(signs[dt,:,i]) for dt in range(lag_max)]
                # candidate = [np.max(signs_0[dt,:,i]) for dt in range(lag_max)]
                nlags[i] = np.argmax(candidate)  # optimal time lag
                ncandidates[i] = np.max(candidate)  # max 0 ratio
            # select top k% candidates and their dts, prune the rest
            k = 0.5
            n_top = k*n_nodes
            top_indices = np.argsort(ncandidates)[-int(n_top):]  # in ncandidates
            # top_values = ncandidates[top_indices]
            top_lags = nlags[top_indices]
        elif isinstance(lag, np.ndarray):
            candidate = np.transpose(signs, (1,2,0))
            # candidate = np.transpose(signs_0, (1,2,0))
            nlags = np.argmax(candidate, axis=2)
            ncandidates = np.max(candidate, axis=2)
            k = 0.5
            n_top = int(k*n_nodes*n_nodes/2)
            top_indices = np.argpartition(ncandidates.flatten(), -n_top)[-n_top:]
            # top_indices = top_indices[np.argsort(ncandidates.flatten()[top_indices])]
            top_indices = np.unravel_index(top_indices, ncandidates.shape)
            top_indices = np.transpose(top_indices)  # [[index0, index1],...]
            top_lags = np.zeros_like(nlags)
            top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]

        
        # use raw array
        time_start = time.perf_counter()
        # graph  = pcmci_raw(data)
        # use patch array
        graph = pcmci(data_bit,nlags=nlags, top_indices=top_indices)
        time_end = time.perf_counter()
        GC_lag = GC*lag
        execute_times.append(time_end-time_start)
        perf.append(compare_graphs(GC, graph))  # tpr, fdr
        lag_perf.append(compare_graphs_lag(GC, GC_lag, top_lags))
    if isinstance(lag, int):
        print("Means and standard deviations for TPR, FDR and AUC with", lag, "time interval")
        print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))
        print("Means and standard deviations for execution time with", lag, "time interval")
        print(np.mean(execute_times), np.std(execute_times))
    elif isinstance(lag, np.ndarray):
        print("Means and standard deviations for TPR, FDR and AUC with lag range in ", lag_range, "time interval")
        print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))
        print("Means and standard deviations for lag accuracy with lag range in ", lag_range, "time interval")
        print(np.mean(lag_perf), np.std(lag_perf))
        print("Means and standard deviations for execution time with lag range in ", lag_range, "time interval")
        print(np.mean(execute_times), np.std(execute_times))
        




