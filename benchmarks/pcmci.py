import numpy as np

import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI


def pcmci(data, nlags=None, t_max=None, t_min=None):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    if nlags is None:
        results = pcmci.run_pcmci(tau_max=5, pc_alpha=None)
    else:
        if t_max is None:
            t_max = np.max(nlags)
        if t_min is None:
            t_min = np.min(nlags)
        # print('t_min, t_max ', t_min, t_max)
        results = pcmci.run_pcmci(tau_min=t_min, tau_max=t_max, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
    q_matrix = (q_matrix < 0.05) * 1  # add 00
    n_nodes = data.shape[1]
    # nlags might be greater than tau_max
    valid_mask = nlags < q_matrix.shape[2]
    results = np.zeros((n_nodes, n_nodes))
    results[valid_mask] = q_matrix[np.arange(n_nodes)[valid_mask], :, nlags[valid_mask]]
    return np.transpose(results)
    # return np.transpose(q_matrix[np.arange(n_nodes), :, nlags])


# raw data, dont know lag beforehand
def pcmci_raw(data):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    # set tau_max manually
    # may induce much bias
    lag_max = int(0.1*data.shape[0])
    results = pcmci.run_pcmci(tau_max=lag_max, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
    q_matrix = (q_matrix < 0.05) * 1  # add 00
    # iterate over all tau to get the best result
    # return columns with the most 1s
    n_nodes = data.shape[1]
    nlags = [0]*n_nodes
    for i in range(n_nodes):
        nlags[i] = np.argmax([np.sum([q_matrix[i,:,t]]) for t in range(lag_max+1)])
    return np.transpose(q_matrix[np.arange(n_nodes), :, nlags])