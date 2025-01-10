import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
from tqdm import *
import time
from utils.synthetic import simulate_er
import matplotlib.pyplot as plt

n_nodes = 10
n_ts = 1000
patch_size = 3
lag_max = int(0.1*n_ts)
lag_range = 3

def draw_plt(tstname, x_list, y_list):
    plt.figure(figsize=(9, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    draw_size = 32
    draw_width = 3
    ax = plt.gca()
    ax.spines['top'].set_linewidth(draw_width)
    ax.spines['right'].set_linewidth(draw_width)
    ax.spines['bottom'].set_linewidth(draw_width)
    ax.spines['left'].set_linewidth(draw_width)
    plt.yticks(fontproperties = 'Arial', size = draw_size)
    plt.xticks(fontproperties = 'Arial', size = draw_size)

    plt.bar(x_list, y_list)
    plt.show()

    plt.savefig(f'../results/motivations/{tstname}/graph.png')


def pcmci_lag_test(data, t_max):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    results = pcmci.run_pcmci(tau_max=t_max, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
    q_matrix = (q_matrix < 0.05) * 1  # add 00
    return q_matrix.shape


def time_lag_efficiency_test(n_nodes, n_ts, lag_range):
    lag = np.random.randint(1, lag_range+1, size=(n_nodes,n_nodes), dtype=int)
    seed = np.random.SeedSequence().generate_state(1)[0]
    data, beta, GC = simulate_er(p=n_nodes, T=n_ts, lag=lag, seed=seed)
    time_consumed = []
    # lags = range(10, 0, -1)
    lags = range(100, 10, -10)
    print('====begin lag range test====')
    for l in tqdm(lags):
        print(f'lag range: {l}')
        time_start = time.perf_counter()
        _shape = pcmci_lag_test(data, t_max=l)
        time_end = time.perf_counter()
        print(f'time cost for lag range {l}(s): {time_end-time_start}')
        time_consumed.append(time_end-time_start)
    tst_dir = '../results/motivations/time_lag_efficiency_test/'
    if not os.path.exists(tst_dir):
        os.makedirs(tst_dir)
    with open(tst_dir+'lists.txt', 'w') as f:
        f.write(f'time consumed: {time_consumed}\n')
        f.write(f'lags: {list(lags)}')
    # draw_plt('time_lag_efficiency_test', x_list=lags, y_list=time_consumed)
    

time_lag_efficiency_test(n_nodes, n_ts, lag_range)
