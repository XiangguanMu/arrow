import pandas as pd
import numpy as np
from bitarray import bitarray
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('/home/yyy/dy/causal-project/utils')
import surd as surd
import matplotlib.pyplot as plt
import pickle
from itertools import combinations as icmb
from itertools import permutations as ipmt

def dataloader(path):
    data = pd.read_csv(path, sep='\t')
    names = data.columns.values
    names = names[1:]  # remove the time column
    n_names = len(names)
    ts = data.iloc[:,0]  # all timestamps
    data = data.iloc[:,1:]  # remove the time column
    data = (data-data.min())/(data.max()-data.min())
    data = np.array(data)
    return data, np.array(ts), names, n_names


def boolList2BinString(lst):
    res = 0
    for i in lst:
        if i:
            res = (res << 1) | 1
        else:
            res = res << 1
    return res


def ternary(n):
    res = ''
    while(n):
        rem = n%3
        res = str(rem)+res
        n //= 3
    return res



# replace np.roll with a faster implementation
# no filling
def roll_array_with_slice(a, shift, axis=0):
    if shift == 0:
        return a

    sl1 = (slice(None),) * axis + (slice(None, -shift),)
    sl2 = (slice(None),) * axis + (slice(-shift, None),)

    # return np.concatenate((np.full(a[sl2].shape, np.nan), a[sl1]), axis=axis)
    return np.concatenate((a[sl2], a[sl1]), axis=axis)


# Load the data
# data: (n_ts, n_nodes)
data, ts, names, n_nodes = dataloader('datasets/DREAM3/simplified/training_data/InSilicoSize10-Ecoli1-trajectories.tsv')
n_ts = ts.shape[0]
patch_size = 3
n_poly = 2  # binomial fitting
n_patches = n_ts - patch_size + 1
nbins = 2
trends = {i:bitarray() for i in range(n_nodes)}
for i in range(n_nodes):
    _data = data[:,i]
    # angles = []
    for j in range(n_patches):
        patch = _data[j:j+patch_size]
        bit0, bit1 = _data[j+1]-_data[j]>0, _data[j+2]-_data[j+1]>0
        trends[i].extend([bit0, bit1])
    # trends[i] = boolList2BinString(trends[i])  # 2*n_patches bits of binary
    # print(bin(trends[names[i]]))  # binary string representation

max_lags = {pmt:0 for pmt in list(ipmt(range(n_nodes), 2)) if pmt[0]!=pmt[1]}
for pmt in max_lags.keys():
    i, itrend = pmt[0], trends[pmt[0]]
    j, jtrend = pmt[1], trends[pmt[1]]
    max_lag_ij = 0
    # offset: iterate over [0, n_patches-1]
    # right offset will not be affected by missing leading zero or non-cyclic rolling
    for dt in range(0, n_patches):
        offset = dt*2
        need_len = (n_patches-dt)*2  # 0s in [0:dt*2] stem from right shift; 0s after dt*2 stem from trend encoding
        xor_int = itrend[-need_len:] ^ jtrend[:need_len]
        # 11->2 opposite 01->1 non 00->0 same
        correlated_sings = [int(xor_int[i])+int(xor_int[i+1]) for i in range(0, need_len, 2)]
        if names[i]=='G3' and names[j]=='G4' and dt<10:
            print(f'==={dt}===')
            # print(correlated_sings)
            # print(f'# of 0: {correlated_sings.count(0)}')
            print(f'# of 1: {correlated_sings.count(1)}')
            # print(f'# of 2: {correlated_sings.count(2)}')


