import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from benchmarks.ngc_model_helper import activation_helper
import time


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        lag_max = np.max(lag)  # size of conv kernel
        layer = nn.Conv1d(num_series, hidden[0], lag_max+1)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          **lag: 2d-array of lags, with shape (num_series, num_series).**
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        # raw data, use default lag size
        if isinstance(lag, int):
            self.networks = nn.ModuleList([
                MLP(num_series, lag, hidden, activation)
                for _ in range(num_series)])
        # patched data, use deduced lag to reduce kernel size
        elif isinstance(lag, np.ndarray):
            self.networks = nn.ModuleList([
                MLP(num_series, lag[i], hidden, activation)
                for i in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC

def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def train_model_ista(cmlp, X, use_raw, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    if use_raw:
        loss = sum([loss_fn(cmlp.networks[i](X[:, :]), X[:, lag:, i:i+1])
                    for i in range(p)])
    else:
        loss = sum([loss_fn(cmlp.networks[i](X[:, :]), X[:, np.max(lag[i]):, i:i+1])
                    for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    epoch_100_times = []
    iter_100 = 0

    for it in range(max_iter):
        
        if it % 100 == 0 and it / 100 == iter_100:
            # print(f'it begin: {it}')
            iter_100 += 1
            epoch_100_time_start = time.perf_counter()

        # Take gradient step.
        smooth.backward()
        for param in cmlp.parameters():
            param.data = param - lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in cmlp.networks:
                prox_update(net, lam, lr, penalty)

        cmlp.zero_grad()

        # Calculate loss for next iteration.
        if use_raw:
            loss = sum([loss_fn(cmlp.networks[i](X[:, :]), X[:, lag:, i:i+1])
                        for i in range(p)])
        else:
            loss = sum([loss_fn(cmlp.networks[i](X[:, :]), X[:, np.max(lag[i]):, i:i+1])
                        for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss + ridge

        if (it+1) % 100 == 0 and (it+1) / 100 == iter_100:
            # print(f'it end: {it}')
            epoch_100_time_end = time.perf_counter()
            # print(epoch_100_time_end-epoch_100_time_start)
            epoch_100_times.append(epoch_100_time_end-epoch_100_time_start)

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list, epoch_100_times

def ngc(data, nlags=None, top_indices=None, use_raw=False, use_constant=False, use_linear=False, use_cp=False):
    n_nodes = data.shape[1]
    if top_indices is not None:
        top_lags = np.zeros_like(nlags)
        top_lags[top_indices[:,0], top_indices[:,1]] = nlags[top_indices[:,0], top_indices[:,1]]
    
    if use_cp:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.tensor(data[np.newaxis], dtype=torch.float32, device=device)
        # select optimal param settings for each case
        if use_linear and use_constant and use_raw:
            max_lag = int(data.shape[0]*0.1)
            cmlp = cMLP(n_nodes, lag=max_lag, hidden=[100]).cuda(device=device)
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=10000, check_every=100, verbose=False)
        elif use_linear and not use_constant and use_raw:
            max_lag = int(data.shape[0]*0.1)
            cmlp = cMLP(n_nodes, lag=max_lag, hidden=[100]).cuda(device=device)
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.001, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=8000, check_every=100, verbose=False)
        elif use_raw: # dream3, non-linear
            max_lag = int(data.shape[0]*0.1)
            cmlp = cMLP(n_nodes, lag=max_lag, hidden=[100]).cuda(device=device)
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.01, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=1500, check_every=100, verbose=False)
        elif use_linear and not use_raw:  # linear + patched
            cmlp = cMLP(n_nodes, lag=top_lags, hidden=[100]).cuda(device=device)
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='GL', max_iter=20000, check_every=100, verbose=False)
        elif not use_linear and not use_raw: # non-linear + patched
            cmlp = cMLP(n_nodes, lag=top_lags, hidden=[100]).cuda(device=device)
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.01, lam_ridge=1e-2, lr=5e-2, penalty='GL', max_iter=5000, check_every=100, verbose=False)
    else:
        X = torch.tensor(data[np.newaxis], dtype=torch.float32)
        if use_raw:
            max_lag = int(data.shape[0]*0.1)
            cmlp = cMLP(n_nodes, lag=max_lag, hidden=[100])
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=20000, check_every=100, verbose=False)
        elif use_linear and not use_raw:  # linear + patched
            cmlp = cMLP(n_nodes, lag=top_lags, hidden=[100])
            train_loss_list, epoch_100_times = train_model_ista(
                cmlp, X, use_raw=use_raw, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='GL', max_iter=20000, check_every=100, verbose=False)
    
    graph = cmlp.GC().cpu().data.numpy()
    if use_raw:
        lag_graph = np.zeros_like(graph).astype(int)
        for i in range(n_nodes):
            # (ks, n)
            GC_est_lag_i = cmlp.GC(ignore_lag=False, threshold=False)[i].cpu().data.numpy().T[::-1]
            # others to i
            lag_i = np.argmax(GC_est_lag_i, axis=0)
            lag_i[np.max(GC_est_lag_i, axis=0) <= 0] = -1  # all weights = 0, no lag
            lag_graph[i] = lag_i+1  # index begins by 0, but lag begins by 1
        
        return graph, lag_graph, epoch_100_times
    elif not use_raw:
        return graph, epoch_100_times