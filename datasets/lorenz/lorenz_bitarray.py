import pandas as pd
import numpy as np
from bitarray import bitarray
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('/home/yyy/dy/causal-project/utils')
from synthetic import simulate_lorenz_96, simulate_lorenz_96_with_delay
import surd as surd
import matplotlib.pyplot as plt
import pickle
from itertools import permutations as ipmt


def boolList2BinString(lst):
    res = 0
    for i in lst:
        if i:
            res = (res << 1) | 1
        else:
            res = res << 1
    return res


def ternary(n,dt):
    res = ''
    while(n):
        rem = n%3
        res = str(rem)+res
        n //= 3
    if(len(res)<n_patches-dt):
        res = '0'*(n_patches-dt-len(res))+res
    return res


# (n_ts, n_node)
n_nodes = 4
n_ts = 1000
patch_size = 3
n_patches = n_ts-patch_size+1
lag = 1
data, GC = simulate_lorenz_96_with_delay(p=n_nodes, T=n_ts, delta_t=0.1, delay_time=lag*0.1)
trends = {i:bitarray() for i in range(n_nodes)}

plt.figure(figsize=(10,5))
for i in range(n_nodes):
    x_i = data[:20,i]
    plt.plot(x_i,label=f'{i+1}')
plt.xticks(range(0, 20, 1))
plt.legend()
plt.grid()
plt.show()
plt.savefig('results/lorenz/trend.jpg')

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
        if i==1 and j==0 and dt==1:
            print(correlated_sings)
            print(f'# of 0: {correlated_sings.count(0)}')
            print(f'# of 1: {correlated_sings.count(1)}')
            print(f'# of 2: {correlated_sings.count(2)}')
            print(f'16: {correlated_sings[16]},17: {correlated_sings[17]}')


