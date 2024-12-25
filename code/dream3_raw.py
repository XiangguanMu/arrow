import pandas as pd
import numpy as np
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('/home/yyy/dy/causal-project/utils')
import surd as surd
import matplotlib.pyplot as plt
import pickle

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

# Load the data
# data: (n_ts, n_nodes)
data, ts, names, n_nodes = dataloader('datasets/DREAM3/simplified/training_data/InSilicoSize10-Ecoli1-trajectories.tsv')
n_ts = ts.shape[0]
patch_size = 3
n_poly = 2  # binomial fitting
n_patches = n_ts - patch_size + 1
dt = 2
nbins = 5

# print(PT_list.shape)

nlags = np.array([1]*n_nodes)  # lag for each node
n_nodes = n_nodes  # use small number to test for now
X = data.T[:n_nodes]  # (n_nodes, n_ts)
plt.figure(figsize=(10,5))
for i in range(n_nodes):
    x_i = data[:,i]
    plt.plot(x_i,label=f'{names[i]}')
plt.legend()
plt.savefig('results/dream3/raw/trend.jpg')

# Storing the results
I_R_results = {}  # Dictionary to store redundant contributions
I_S_results = {}  # Dictionary to store synergistic contributions
MI_results = {}   # Dictionary to store mutual information results
info_leak_results = {}  # Dictionary to store information leak results

# todo: mapping id to names

import pymp
import it_tools as it
from scipy.stats import binned_statistic_dd
if __name__ == '__main__':

    for i in range(n_nodes):
        print(f'SURD CAUSALITY FOR SIGNAL {names[i]}')

        # Organize data (0 target variable, 1: agent variables)
        # agent variables includes the target variable itself
        # Y = np.vstack([np.roll(X[i, nlags[i]:], nlags[i]), X[:, :-nlags[i]]])
        Y = np.vstack([X[i, nlags[i]:], X[:, :-nlags[i]]])

        # Run SURD
        hist, _ = np.histogramdd(Y.T, nbins)  # bottleneck 1?
        I_R, I_S, MI, info_leak = surd.surd(hist)
        # # Run SURD hd
        # I_R, I_S, MI = surd.surd_hd(Y, nbins, max_combs=3)
        # # Calculate information leak
        # hist = it.myhistogram(Y[0,:].T, nbins)
        # H  = it.entropy_nvars(hist, (0,) )
        # info_leak = 1 - (sum(I_R.values()) + sum(I_S.values())) / H

        # Print results
        # surd.nice_print(I_R, I_S, MI, info_leak)
        # print('\n')

        # Save the results
        I_R_results[i+1] = I_R
        I_S_results[i+1] = I_S
        MI_results[i+1] = MI
        info_leak_results[i+1] = info_leak

        # Plot results
        fig, axs = plt.subplots(1, 2, figsize=(18, 4), gridspec_kw={'width_ratios': [10, 1]})
        surd.plot(I_R, I_S, info_leak, axs, n_nodes, threshold=1e-10)
        axs[0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {names[i]}}} / I \\left({names[i]}^+ ; \\mathrm{{\\mathbf{{G}}}} \\right)$', pad=12)
        axs[1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {names[i]}}}}}{{H \\left({names[i]} \\right)}}$', pad=20)
        # plt.tight_layout(w_pad=-13, h_pad=0)
        # plt.show()
        plt.savefig(f'results/dream3/raw/{names[i]}.jpg')

    # Save the results to a file
    with open('results/sz10-ecoli1.pkl', 'wb') as file:
        pickle.dump({
            'I_R_results': I_R_results,
            'I_S_results': I_S_results,
            'MI_results': MI_results,
            'info_leak_results': info_leak_results
        }, file)

    print("Results saved to 'results/sz10-ecoli1.pkl'")


