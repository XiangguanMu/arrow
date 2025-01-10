import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
from bitarray import bitarray
from itertools import permutations as ipmt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def estimate_lags(data, patch_size, lag, lag_max):
    """
    Estimate lags for each pair of nodes
    Input:
        data (np.ndarray): (n_ts, n_node), input time series data.
        patch_size (int): Each patch contains patch_size time points.
        lag (np.nadarray): lag[i][j] is the lag from j to i.
        lag_max (int): The upper bound of lag to be considered.
    Output:
        nlags (np.ndarray): (n_node, n_node), optimal time lags for each pair of nodes.
    """
    n_ts, n_nodes = data.shape
    n_patches = n_ts-patch_size+1
    trends = {i:bitarray() for i in range(n_nodes)}
    data_bit = np.zeros((n_patches,n_nodes))
    for i in range(n_nodes):
        _data = data[:,i]
        for j in range(n_patches):
            patch = _data[j:j+patch_size]
            bit0, bit1 = _data[j+1]-_data[j]>0, _data[j+2]-_data[j+1]>0
            trends[i].extend([bit0, bit1])
            data_bit[j][i] = bit0*2+bit1  # 00->0 01->1 10->2 11->3, 4 bins
    pair_list = list(ipmt(range(n_nodes), 2))
    pair_list.extend([(i,i) for i in range(n_nodes)])
    # pair_list = [p for p in pair_list if p[0]!=p[1]]
    signs_0 = np.zeros((lag_max, n_nodes, n_nodes))
    signs_1 = np.zeros((lag_max, n_nodes, n_nodes))
    signs_2 = np.zeros((lag_max, n_nodes, n_nodes))
    for pmt in pair_list:
        i, itrend = pmt[0], trends[pmt[0]]
        j, jtrend = pmt[1], trends[pmt[1]]
        max_lag_ij = 0
        # offset: iterate over [0, n_patches-1]
        # right offset will not be affected by missing leading zero or non-cyclic rolling
        for dt in range(1, n_patches):
        # for dt in range(0, n_patches):
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
        
    # (n_nodes, n_nodes, lag_max)
    candidate = np.transpose(signs, (1,2,0))
    nlags = np.argmax(candidate, axis=2)
    ncandidates = np.max(candidate, axis=2)
    k = 0.5
    n_top = int(k*n_nodes*n_nodes/2)
    
    # # no threshold
    # top_indices = np.argpartition(ncandidates.flatten(), -n_top)[-n_top:]
    # top_indices = np.unravel_index(top_indices, ncandidates.shape)
    # top_indices = np.transpose(top_indices)  # [[index0, index1],...]

    # set threshold
    flag_indices = np.argsort(ncandidates.ravel())[::-1]
    sorted_values = ncandidates.ravel()[flag_indices]
    sorted_indices = np.array(np.unravel_index(flag_indices, ncandidates.shape)).T
    top_indices = []
    for i, (val, idx) in enumerate(zip(sorted_values, sorted_indices)):
        if val>0.33:
            top_indices.append(idx)
        if len(top_indices)==n_top:
            break
    top_indices = np.array(top_indices)
    # # debug mode
    # top_indices = np.argpartition(ncandidates.flatten(), -n_top)[-n_top:]
    # top_indices = np.unravel_index(top_indices, ncandidates.shape)
    # top_indices = np.transpose(top_indices)  # [[index0, index1],...]
    # top_lags = np.zeros_like(nlags)
    # top_candidates = np.zeros_like(ncandidates)
    # top_candidates[top_indices[:,0], top_indices[:,1]] = ncandidates[top_indices[:,0], top_indices[:,1]]
    # top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
    return data_bit, nlags, top_indices

