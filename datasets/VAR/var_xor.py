import pandas as pd
import numpy as np
from bitarray import bitarray
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('/home/yyy/dy/causal-project/utils')
from synthetic import simulate_var
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
n_nodes = 3
n_ts = 1000
patch_size = 3
n_patches = n_ts-patch_size+1
lag = 1
data, beta, GC = simulate_var(p=n_nodes, T=n_ts, lag=lag)
trends = {i:[] for i in range(n_nodes)}

plt.figure(figsize=(10,5))
for i in range(n_nodes):
    x_i = data[:20,i]
    plt.plot(x_i,label=f'{i+1}')
plt.xticks(range(0, 20, 1))
plt.legend()
plt.grid()
plt.show()
plt.savefig('results/var/trend.jpg')

for i in range(n_nodes):
    _data = data[:,i]
    # angles = []
    for j in range(n_patches):
        patch = _data[j:j+patch_size]
        bit0, bit1 = _data[j+1]-_data[j]>0, _data[j+2]-_data[j+1]>0
        trends[i].extend([bit0, bit1])
        if j==17:
            print(f'patch: {patch}, bit0: {bit0}, bit1: {bit1}')
    trends[i] = boolList2BinString(trends[i])  # 2*n_patches bits of binary
    # print(bin(trends[names[i]]))  # binary string representation

max_lags = {pmt:0 for pmt in list(ipmt(range(n_nodes), 2)) if pmt[0]!=pmt[1]}
for pmt in max_lags.keys():
    i, itrend = pmt[0], trends[pmt[0]]
    j, jtrend = pmt[1], trends[pmt[1]]
    max_lag_ij = 0
    if i==0 and j==1:
        print('i,j: ',i, j)
    # offset: iterate over [0, n_patches-1]
    # right offset will not be affected by missing leading zero or non-cyclic rolling
    for dt in range(0, n_patches):
        offset = dt*2
        xor_int = itrend ^ (jtrend>>offset)
        need_len = (n_patches-dt)*2  # 0s in [0:dt*2] stem from right shift; 0s after dt*2 stem from trend encoding
        bit_mask = (1 << need_len) - 1  # only perserve the final `need_len` bits
        act_int = xor_int & bit_mask  # preserved integer
        act_len = act_int.bit_length()  # actual length of preserved integer might be less than need_len
        # find the max length of consecutive correlated patches
        flag = 0  # whether or not in a consecutive cp
        max_lag_ij_dt = 0
        correlated_signs = 0
        for p_i in range(0, (act_len-act_len%2)//2):
            pair = act_int & 0b11
            is_correlated = 1-((pair & 0b01) ^ ((pair & 0b10) >> 1))  # 1: correlated
            # correlated_signs = (correlated_signs<<1) | is_correlated
            # 01/10: none,0
            if not is_correlated:
                correlated_signs *= 3
            # 00: same,1
            elif not pair:  
                correlated_signs = correlated_signs*3 + 1
            # 11: oppo,2
            elif pair:
                correlated_signs = correlated_signs*3 + 2
            act_int >>= 2
        # leading 0s here stem from trend encoding
        for p_i in range(0, (need_len-(act_len+act_len%2))//2):
            # 00: same, 1
            correlated_signs = correlated_signs*3 + 1
        if i==1 and j==0 and dt==0:
            print(bin(trends[i]))
            print(bin(trends[j]))
            print('i,j,dt: correlated ',i, j, dt, ternary(correlated_signs,dt)[:20])

