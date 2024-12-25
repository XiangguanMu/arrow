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
from itertools import permutations as ipmt
np.random.seed(10)

def mediator(N,lag):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-nlag):
        q1[n+lag] = np.sin(q2[n]) + 0.001*W1[n]
        q2[n+lag] = np.cos(q3[n]) + 0.01*W2[n]
        q3[n+lag] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

n_nodes = 3
n_ts = 1100            # Number of time steps to perform the integration of the system
samples = n_ts-100      # Number of samples to be considered (remove the transients)
nbins = 2              # Number of bins to disctrize the histogram
nlag = 3              # Time lag to perform the causal analysis
patch_size = 3
n_patches = samples-patch_size+1

file_path = f'datasets/tri/mediator_nlag_{nlag}.npy'
if os.path.isfile(file_path):
    data = np.load(file_path)
else:
    qs = mediator(n_ts,nlag)
    data = np.array([q[-samples:] for q in qs])
    np.save(file_path,data)
data = data.T
trends = {i:bitarray() for i in range(n_nodes)}

plt.figure(figsize=(10,5))
for i in range(n_nodes-1):
    x_i = data[:20,i]
    plt.plot(x_i,label=f'q{i+1}')
plt.xticks(range(0, 20, 1))
plt.legend()
plt.grid()
plt.show()
plt.savefig('results/tri/trend.jpg')

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
        if i==0 and j==1 and dt==2:
            print(correlated_sings)
            print(f'# of 0: {correlated_sings.count(0)}')
            print(f'# of 1: {correlated_sings.count(1)}')
            print(f'# of 2: {correlated_sings.count(2)}')


