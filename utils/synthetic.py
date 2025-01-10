import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import roc_auc_score
from functools import partial

noise_names = ['gaussian', 'uniform', 'exponential', 'gamma']
function_names = ['linear', 'sqrt', 'sin', 'tanh']

def erdos_renyi(n, p):
    '''Generate Erdos-Renyi random graph.'''
    fully_connect = np.triu(np.ones((n, n)), k=1)
    rand_mask = np.random.binomial(n=1, p=p, size=(n, n))
    adjmatrix = fully_connect * rand_mask
    return adjmatrix


def symmtric_exp(beta,size):
    # beta large => std large
    # by default, beta=0.5 (similar to N(0,1))
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    exp = np.random.exponential(scale=beta,size=size)
    return randsign*exp

def symmetric_gamma(k,theta,size):
    # please fix k=2, and theta large => std large
    # by default, theta=0.3 (similar to N(0,1))
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    gamma = np.random.gamma(shape=k,scale=theta,size=size)
    return randsign*gamma

def distribution(name,std=1,beta=0.5,k=2,theta=0.3):
    # usage: distribution(name)(size=size)
    if name=='gaussian':
        funct = partial(np.random.normal,loc=0,scale=std)
    elif name=='uniform':
        funct = partial(np.random.uniform,low=-std,high=std)
    elif name=='exponential':
        funct = partial(symmtric_exp,beta=beta)
    elif name=='gamma':
        funct = partial(symmetric_gamma,k=k,theta=theta)
    else:
        raise ValueError('Only support gaussian,exponential,gamma distribution.')
    return funct

def function(name):
    # usage: y = function(name)(x)
    if name=='linear':
        funct = lambda x:x
    elif name=='sqrt':
        funct = lambda x: np.sign(x)*np.sqrt(abs(x)/2)
    elif name=='sin':
        funct = lambda x:np.sin(np.pi/2*x)
    elif name=='tanh':
        funct = lambda x:np.tanh(x)
    else:
        raise ValueError('Only support linear,sqrt,sin,tanh function.')
    return lambda x:funct(x)


# avoid infinite growing or decaying of the results
def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)  # decrease beta recursively
    else:
        return beta


# def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=1e-4, seed=0):
#     if seed is not None:
#         np.random.seed(seed)

#     # Set up coefficients and Granger causality ground truth.
#     GC = np.eye(p, dtype=int)  # diagonal matrix
#     # GC = np.zeros((p,p), dtype=int)  # diagonal matrix
#     beta = np.eye(p) * beta_value  # diagonal matrix too
#     # beta = np.zeros((p,p))  # diagonal matrix too

#     # num_nonzero = 1
#     num_nonzero = int(p * sparsity) - 1
#     for i in range(p):
#         choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
#         choice[choice >= i] += 1
#         beta[i, choice] = beta_value
#         GC[i, choice] = 1

#     beta = np.hstack([beta for _ in range(lag)])
#     beta = make_var_stationary(beta)

#     # Generate data.
#     burn_in = 100
#     errors = np.random.normal(scale=sd, size=(p, T + burn_in))
#     X = np.zeros((p, T + burn_in))
#     X[:, :lag] = errors[:, :lag]
#     for t in range(lag, T + burn_in):
#         X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
#         X[:, t] += errors[:, t-1]

#     return X.T[burn_in:], beta, GC


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    # num_nonzero = 1
    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    # beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    # X[:, :lag] = errors[:, :lag]
    X[:, :np.max(lag)] = errors[:, :np.max(lag)]
    parents = {i: np.where(GC[i, :] == 1)[0] for i in range(p)}
    # X[:, :lag] = errors[:, :lag]
    for t in range(np.max(lag), T + burn_in):
        X[:, t] = np.random.normal(scale=sd, size=p)
        for i in range(p):
            for j in parents[i]:
                X[i, t] += np.dot(beta[i, j], X[j, t-lag[i][j]])
        # X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        # X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def simulate_er(p, T, lag, sparsity=0.4, beta_value=1.0, sd=1e-4, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = erdos_renyi(p, sparsity)

    beta = make_var_stationary(GC)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :np.max(lag)] = errors[:, :np.max(lag)]
    parents = {i: np.where(GC[i, :] == 1)[0] for i in range(p)}
    n_noise = len(noise_names)
    n_funcs = len(function_names)
    for t in range(np.max(lag), T + burn_in):
        i_noise = np.random.randint(t)%n_noise
        X[:, t] = distribution(noise_names[i_noise])(size=p)
        for i in range(p):
            for j in parents[i]:
                i_func = np.random.randint(n_funcs)%n_funcs
                X[i, t] += function(function_names[i_func])(X[j, t-lag[i][j]])


    return X.T[burn_in:], beta, GC



def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    dxdt = [(x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F for i in range(p)]

    return dxdt
    

def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.01, burn_in=100,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC

delay_buffer = None

def lorenz_with_delay(x, t, F, delay_buffer, delay_steps, delta_t):
    '''Partial derivatives for Lorenz-96 ODE with time delay.'''
    p = len(x)
    dxdt = np.zeros(p)

    delayed_x = delay_buffer[delay_steps]

    dxdt = [(delayed_x[(i+1)%p]-delayed_x[(i-2)%p])*delayed_x[(i-1)%p]-delayed_x[i]+F for i in range(p)]

    return np.array(dxdt)

def simulate_lorenz_96_with_delay(p, T, F=10.0, delta_t=1e-3, sd=0.01, burn_in=100, delay_time=1.0, seed=0):
    if seed is not None:
        np.random.seed(seed)
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    delay_steps = int(delay_time / delta_t)
    delay_buffer = [x0.copy()]
    for _ in range(delay_steps):
        delay_buffer.append(np.random.normal(scale=0.01, size=p))

    def custom_ode_solver(f, x0, t, args):
        nonlocal delay_buffer
        result = [x0.copy()]
        for i in range(1, len(t)):
            current_x = result[-1]
            dxdt = f(current_x, t[i], *args, delay_buffer, delay_steps, delta_t)
            new_x = current_x + 1e-3 * dxdt * (t[i] - t[i - 1])  # 使用欧拉法（不收敛）
            result.append(new_x)
            delay_buffer = [new_x] + delay_buffer[:-1]
        return np.array(result)

    X = custom_ode_solver(lorenz_with_delay, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=X.shape)
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def compare_graphs(true_graph, estimated_graph):
    """Compute performance measures on (binary) adjacency matrix
    Input:
     - true_graph: (dxd) np.array, the true adjacency matrix
     - estimated graph: (dxd) np.array, the estimated adjacency matrix (weighted or unweighted)
    """

    def structural_hamming_distance(W_true, W_est):
        """Computes the structural hamming distance"""

        pred = np.flatnonzero(W_est != 0)
        cond = np.flatnonzero(W_true)
        cond_reversed = np.flatnonzero(W_true.T)

        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

        pred_lower = np.flatnonzero(np.tril(W_est + W_est.T))
        cond_lower = np.flatnonzero(np.tril(W_true + W_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        return shd

    num_edges = len(true_graph[np.where(true_graph != 0.0)])

    tam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in true_graph])
    eam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in estimated_graph])

    tp = len(np.argwhere((tam + eam) == 2))
    fp = len(np.argwhere((tam - eam) < 0))
    tn = len(np.argwhere((tam + eam) == 0))
    fn = num_edges - tp
    x = [tp, fp, tn, fn]

    if x[0] + x[1] == 0:
        precision = 0
    else:
        precision = float(x[0]) / float(x[0] + x[1])
    if tp + fn == 0:
        tpr = 0
    else:
        tpr = float(tp) / float(tp + fn)
    if x[2] + x[1] == 0:
        specificity = 0
    else:
        specificity = float(x[2]) / float(x[2] + x[1])
    if precision + tpr == 0:
        f1 = 0
    else:
        f1 = 2 * precision * tpr / (precision + tpr)
    if fp + tp == 0:
        fdr = 0
    else:
        fdr = float(fp) / (float(fp) + float(tp))
    if tn + fp == 0:
        fpr = 0
    else:
        fpr = float(fp) / (float(tn) + float(fp))

    shd = structural_hamming_distance(true_graph, estimated_graph)

    AUC = roc_auc_score(true_graph.flatten(), estimated_graph.flatten())

    return tpr, fpr, AUC
    # return tpr, fdr, AUC


def compare_graphs_lag(true_graph, true_graph_lag, estimated_graph_lag):
    """
    Compute the ratio of all true edges w./w.o. their true lags are detected
    """
    # TPR: # true lag detectes / # true edges
    num_edges = len(true_graph[np.where(true_graph != 0.0)])
    true_idxs = np.nonzero(true_graph)
    estimated_edges = np.array([true_graph_lag[true_idxs[0][i]][true_idxs[1][i]]==estimated_graph_lag[true_idxs[0][i]][true_idxs[1][i]] for i in range(num_edges)])
    num_edges_correct_estimated = np.sum(estimated_edges)
    tpr = num_edges_correct_estimated/num_edges

    # FPR: # false lag detects / # empty lags
    false_idxs = np.nonzero(1-true_graph)
    # detect lag where there is no edge
    estimated_zeros = np.array([true_graph_lag[false_idxs[0][i]][false_idxs[1][i]]!=estimated_graph_lag[false_idxs[0][i]][false_idxs[1][i]] for i in range(len(false_idxs[0]))])
    num_zeros_correct_estimated = np.sum(estimated_zeros)
    fpr = num_zeros_correct_estimated/len(false_idxs[0])

    estimated_graph_lag_bool = np.zeros_like(estimated_graph_lag)
    estimated_graph_lag_bool[true_idxs] = (estimated_graph_lag[true_idxs]==true_graph_lag[true_idxs])
    AUC = roc_auc_score(true_graph.flatten(), estimated_graph_lag_bool.flatten())
    return tpr, fpr, AUC