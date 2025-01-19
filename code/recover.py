import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import numpy as np
from utils.synthetic import compare_graphs, compare_graphs_lag

raw_dir = '../results/nps/ngc_multiple_raw_10_'
for i in [3,5,7,9,15,20]:
    perf = []
    lag_perf = []
    execute_times = []
    dir = raw_dir+str(i)+'/'
    est_lag_dir = dir+'estimate_lag/'
    gc_dir = dir+'GC/'
    gc_lag_dir = dir+'GC_lag/'
    graph_dir = dir+'graph/'
    dur_dir = dir+'duration/'

    for j in range(10):
        est_lag_j = np.load(est_lag_dir+str(j)+'.npy')
        gc_j = np.load(gc_dir+str(j)+'.npy')
        gc_lag_j = np.load(gc_lag_dir+str(j)+'.npy')
        graph_j = np.load(graph_dir+str(j)+'.npy')
        duration_j = np.load(dur_dir+str(j)+'.npy')
        perf.append(compare_graphs(gc_j, graph_j))
        lag_perf.append(compare_graphs_lag(gc_j, gc_lag_j, est_lag_j))
        execute_times.append(duration_j)
    
    print('=====')
    print("Means and standard deviations for TPR, FPR and AUC with lag range in ", i, "time interval")
    print(np.mean(np.reshape(perf, (-1, 3)), axis=0), np.std(np.reshape(perf, (-1, 3)), axis=0))

    print("Means and standard deviations for lag TPR, FPR and AUC with lag range in ", i, "time interval")
    print(np.mean(np.reshape(lag_perf, (-1, 3)), axis=0), np.std(np.reshape(lag_perf, (-1, 3)), axis=0))

    print("Means and standard deviations for execution time with lag range in ", i, "time interval")
    print(np.mean(execute_times), np.std(execute_times))
