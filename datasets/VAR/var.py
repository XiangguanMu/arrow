import sys
sys.path.append('/home/yyy/dy/causal-project/utils')
from synthetic import simulate_var
import numpy as np
import surd as surd
import pymp
import it_tools as it
import matplotlib.pyplot as plt
import networkx as nx

# (n_ts, n_node)
n_nodes = 3
lag = 1
X_np, beta, GC = simulate_var(p=n_nodes, T=1000, lag=lag)
X = X_np.T 

# plt.figure(figsize=(10,5))
# for i in range(n_nodes):
#     x_i = X_np[:100,i]
#     plt.plot(x_i,label=f'{i+1}')
# plt.legend()
# plt.show()
# plt.savefig('results/var/trend.jpg')

nlags = [lag]*n_nodes
nbins = 2

# Storing the results
I_R_results = {}  # Dictionary to store redundant contributions
I_S_results = {}  # Dictionary to store synergistic contributions
MI_results = {}   # Dictionary to store mutual information results
info_leak_results = {}  # Dictionary to store information leak results

# # todo: mapping id to names

G = nx.DiGraph()
node2name = {i: f'${i+1}$' for i in range(n_nodes)}
edges = np.argwhere(GC).tolist()
edges = [(e[1], e[0]) for e in edges]
G.add_nodes_from(node2name)
G.add_edges_from(edges)
plt.figure(figsize=(8, 8))
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='skyblue')
nx.draw_networkx_labels(G, pos, labels=node2name)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.savefig('results/var/graph.png')

if __name__ == '__main__':

    for i in range(n_nodes):
        print(f'SURD CAUSALITY FOR SIGNAL {i+1}')

        # Organize data (0 target variable, 1: agent variables)
        # agent variables includes the target variable itself
        if lag==0:
            Y = np.vstack([X[i, :], X[:, :]])
        else:
            Y = np.vstack([X[i, nlags[i]:], X[:, :-nlags[i]]])
        # Y = np.vstack([np.roll(X[i, nlags[i]:], nlags[i]), X[:, :-nlags[i]]])  # offset time lag?

        # Run SURD
        hist, _ = np.histogramdd(Y.T, nbins)
        I_R, I_S, MI, info_leak = surd.surd(hist)
        # # Run SURD hd
        # I_R, I_S, MI = surd.surd_hd(Y, nbins, max_combs=3)
        # # Calculate information leak
        # hist = it.myhistogram(Y[0,:].T, nbins)
        # H  = it.entropy_nvars(hist, (0,) )
        # info_leak = 1 - (sum(I_R.values()) + sum(I_S.values())) / H

        # Save the results
        I_R_results[i+1] = I_R
        I_S_results[i+1] = I_S
        MI_results[i+1] = MI
        info_leak_results[i+1] = info_leak

        # Plot results
        fig, axs = plt.subplots(1, 2, figsize=(18, 4), gridspec_kw={'width_ratios': [10, 1]})
        surd.plot(I_R, I_S, info_leak, axs, n_nodes, threshold=1e-10)
        axs[0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left({i+1}^+ ; \\mathrm{{\\mathbf{{G}}}} \\right)$', pad=12)
        axs[1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left({i+1} \\right)}}$', pad=20)
        # plt.tight_layout(w_pad=-13, h_pad=0)
        plt.savefig(f'results/var/{i+1}.png')