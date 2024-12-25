import pandas as pd
import numpy as np
from bitarray import bitarray
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from utils.synthetic import simulate_var,simulate_var_one_slice
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
lag = 2
data, beta, GC = simulate_var_one_slice(p=n_nodes, T=n_ts, lag=lag)
# data, beta, GC = simulate_var(p=n_nodes, T=n_ts, lag=lag)
trends = {i:bitarray() for i in range(n_nodes)}

draw_x = 50
plt.figure(figsize=(10,5))
for i in range(n_nodes):
    x_i = data[:draw_x,i]
    plt.plot(x_i,label=f'{i+1}')
plt.xticks(range(0, draw_x, 1))
plt.legend()
plt.grid()
# plt.show()
plt.savefig('../../results/var/trend.jpg')

import networkx as nx
G = nx.DiGraph()
node2name = {i: f'${i+1}$' for i in range(n_nodes)}
G.add_nodes_from(node2name)
edges = np.argwhere(GC).tolist()
edges = [(e[1], e[0]) for e in edges]
G.add_edges_from(edges)
plt.figure(figsize=(8, 8))
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='skyblue')
nx.draw_networkx_labels(G, pos, labels=node2name)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.savefig('../../results/var/graph.png')

for i in range(n_nodes):
    _data = data[:,i]
    for j in range(n_patches):
        patch = _data[j:j+patch_size]
        bit0, bit1 = _data[j+1]-_data[j]>0, _data[j+2]-_data[j+1]>0
        trends[i].extend([bit0, bit1])

max_lags = {pmt:0 for pmt in list(ipmt(range(n_nodes), 2)) if pmt[0]!=pmt[1]}
pcmci_dt = 0
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
        if (correlated_sings.count(0)/len(correlated_sings))>0.8:
            pcmci_dt = dt
            break
        if i==0 and j==1 and dt==lag:
            print(correlated_sings)
            print(f'# of 0: {correlated_sings.count(0)}')
            print(f'# of 1: {correlated_sings.count(1)}')
            print(f'# of 2: {correlated_sings.count(2)}')


