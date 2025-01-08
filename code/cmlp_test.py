import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import torch
import numpy as np
from tqdm import *
from utils.synthetic import simulate_er, compare_graphs, compare_graphs_lag,simulate_var
from benchmarks.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
import matplotlib.pyplot as plt

# Simulate data
n_nodes = 10
lag = np.ones((n_nodes,n_nodes), dtype=int)*1
# X_np, beta, GC = simulate_er(p=n_nodes, T=1000, lag=lag)
X_np, beta, GC = simulate_var(p=10, T=1000, lag=lag)
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32)
print(X.shape)
# cmlp = cMLP(X.shape[-1], lag=lag, hidden=[100])
# # Train with ISTA
# # train_loss_list = train_model_ista(
# #     cmlp, X, lam=0.004625, lam_ridge=1e-2, lr=5e-2, penalty='GSGL', max_iter=3000,
# #     check_every=100)
# train_loss_list = train_model_ista(
#     cmlp, X, lam=0.01, lam_ridge=1e-2, lr=5e-2, penalty='GL', max_iter=5000,
#     check_every=100)
# # Verify learned Granger causality
# GC_est = cmlp.GC().cpu().data.numpy()

# print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
# print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
# print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
axarr[0].imshow(GC, cmap='Blues')
axarr[0].set_title('GC actual')
axarr[0].set_ylabel('Affected series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_title('GC estimated')
axarr[1].set_ylabel('Affected series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.show()

