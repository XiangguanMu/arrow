import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
from causallearn.search.FCMBased import lingam


def varlingam(data, nlags=None, top_indices=None, use_raw=False, use_constant=False):
    n_ts, n_nodes = data.shape
    lag_max = int(0.1*n_ts)
    if use_raw:
        model = lingam.VARLiNGAM(lags=80, criterion='bic', prune=True)
        # model = lingam.VARLiNGAM(lags=lag_max, criterion='bic', prune=True)  # error
        model.fit(data)
        weighted_graph = np.sum(model.adjacency_matrices_, axis=0)/model.adjacency_matrices_.shape[0]
        graph = np.where(weighted_graph, 1, 0)
        # lag_graph: contribute most to graph
        lag_graph = np.argmax(model.adjacency_matrices_, axis=0)
        return graph, lag_graph
    elif not use_raw:
        t_max = np.max(nlags[top_indices[:,0], top_indices[:,1]])
        # Avoid using default prune strategy to speed up
        model = lingam.VARLiNGAM(lags=t_max, criterion=None)
        model.fit(data)
        matrices = model.adjacency_matrices_
        valid_mask = nlags < matrices.shape[0]
        weighted_graph = np.zeros((n_nodes, n_nodes))
        rows, cols = np.where(valid_mask)
        weighted_graph[rows,cols] = matrices[nlags[rows,cols], rows, cols]
        graph = np.where(weighted_graph>0.1, 1, 0)
        return graph


