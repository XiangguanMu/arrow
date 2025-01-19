import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
tig_pth = os.path.join(cur_dir, 'tigramite')
sys.path.append(tig_pth)
import numpy as np

import data_processing as pp
from independence_tests import ParCorr
from pcmci import PCMCI


def pcmci(data, nlags=None, top_indices=None, use_raw=False, use_constant=False):
    n_nodes = data.shape[1]
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    if use_raw == True:
        lag_max = int(0.1*data.shape[0])
        results = pcmci.run_pcmci(tau_max=lag_max, pc_alpha=None)
        q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
        q_matrix = (q_matrix < 0.05) * 1  # add 00
        # iterate over all tau to get the best result
        # return columns with the most 1s
        nlags = [0]*n_nodes
        for i in range(n_nodes):
            nlags[i] = np.argmax([np.sum([q_matrix[i,:,t]]) for t in range(lag_max+1)])
        estimated_graph = np.transpose(q_matrix[np.arange(n_nodes), :, nlags])
        estimated_lag = np.zeros_like(estimated_graph)
        rows, cols = np.where(estimated_graph>0)
        for i in range(len(rows)):
            estimated_lag[rows[i], cols[i]] = nlags[cols[i]]
        # estimated_lag[rows, cols] = nlags[cols]  # why error
        return estimated_graph, estimated_lag
    elif use_raw == False:
        if isinstance(nlags, np.ndarray):
            t_max = np.max(nlags[top_indices[:,0], top_indices[:,1]])
            t_min = np.min(nlags[top_indices[:,0], top_indices[:,1]])
            # print('t_min, t_max ', t_min, t_max)

            patch_selected_links = {}
            # for deduced pairs, use deduced lags
            for edge in top_indices:
                if edge[0] not in patch_selected_links.keys():
                    patch_selected_links[edge[0]] = [
                        (edge[1], -nlags[edge[0], edge[1]])
                    ]
                else:
                    patch_selected_links[edge[0]].extend([
                        (edge[1], -nlags[edge[0], edge[1]])
                    ])
            # for the rest, use all lags to avoid missing
            for i in range(n_nodes):
                if i not in patch_selected_links.keys():
                    patch_selected_links[i] = [(var, -lag)
                        for var in range(n_nodes)
                        for lag in range(t_min, t_max+1)
                    ]
                
            results = pcmci.run_pcmci(tau_min=t_min, tau_max=t_max, selected_links=patch_selected_links, pc_alpha=None)
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
            q_matrix = (q_matrix < 0.05) * 1  # add 00
            n_nodes = data.shape[1]
            valid_mask = nlags < q_matrix.shape[2]
            results = np.zeros((n_nodes, n_nodes))
            rows, cols = np.where(valid_mask)
            results[rows, cols] = q_matrix[cols, rows, nlags[rows,cols]]
            return results