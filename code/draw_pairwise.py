import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# calculated the area between two curves after offseting in y axis
# avg_i, avg_j have been offsetted before passing into this function
def calculate_area(coef_i, delta_i, avg_i, coef_j, delta_j, avg_j, fig, t):
    n_line = len(delta_i)  # 2 lines for each patch in this case
    ki = []
    bi = []
    kj = []
    bj = []
    for k in range(n_line):
        x_k = avg_i[k]
        y_ik = np.polyval(coef_i, x_k)
        k_ik = delta_i[k]
        b_ik = y_ik - k_ik*x_k
        # print(f'k_i{k}, b_i{k}',k_ik, b_ik)
        ki.append(k_ik)
        bi.append(b_ik)

        y_jk = np.polyval(coef_j, x_k)
        k_jk = delta_j[k]
        b_jk = y_jk - k_jk*x_k
        # print(f'k_j{k}, b_j{k}',k_jk, b_jk)
        kj.append(k_jk)
        bj.append(b_jk)
    # plt.figure(figsize=(5,5))
    # draw the two lines of i and j
    sub = fig.add_subplot(1,19,t+1)
    linelen = 1.0
    for k in range(n_line):
        xi = np.linspace(avg_i[k]-linelen, avg_i[k]+linelen, 100)
        yi = ki[k]*xi+bi[k]
        xj = np.linspace(avg_j[k]-linelen, avg_j[k]+linelen, 100)
        yj = kj[k]*xj+bj[k]
        sub.plot(xi, yi, color='green', linestyle='--')
        sub.plot(xj, yj, color='blue', linestyle='--')
    



# Load the data
data, ts, names, n_names = dataloader('../datasets/DREAM3/simplified/training_data/InSilicoSize10-Ecoli1-trajectories.tsv')
patch_size = 3
n_poly = 2  # binomial fitting
scale = np.array(ts)[1]-np.array(ts)[0]
n_ts = len(data[:,0])
n_patches = n_ts-patch_size+1
PTdict_list = []

for i in range(n_names):
    name = names[i]
    _data = data[:,i]
    _data = [d*1 for d in _data]
    coefficients = []
    deltas = []
    averages = []
    
    # iterate patch
    for j in range(n_patches):
        patch = _data[j:j+patch_size]
        coef = np.polyfit(np.arange(patch_size), patch, n_poly)
        d_coef = np.polyder(coef)
        delt = []
        avg = []
        for k in range(patch_size-1):  # k=0,1
            xk = (ts[j+k]+ts[j+k+1])/2  # average of two consecutive points
            xk /= scale  # scale to align with the x axis
            avg.append(xk)
            d_x = np.polyval(d_coef, xk-j)
            delt.append(d_x)
        deltas.append(delt)
        coefficients.append(coef)
        averages.append(avg)
    PTdict_list.append({'name': name, 'coefficients': coefficients, 'deltas': deltas, 'averages': averages})

# print(PTdict_list[0]['averages'])


# use G0 and G1 for test
# for t in range(n_patches):
#     for i, j in combinations(range(n_names),2):
for i, j in combinations(range(n_names),2):
    fig = plt.figure(figsize=(18,5))
    for t in range(n_patches):
        name_i = names[i]
        name_j = names[j]
        coef_i = PTdict_list[i]['coefficients'][t]
        coef_j = PTdict_list[j]['coefficients'][t]
        delta_i = PTdict_list[i]['deltas'][t]
        delta_j = PTdict_list[j]['deltas'][t]
        avg_i = [a-t for a in PTdict_list[i]['averages'][t]]  # offset
        avg_j = [a-t for a in PTdict_list[j]['averages'][t]]
        # print(avg_i, avg_j)
        calculate_area(coef_i, delta_i, avg_i, coef_j, delta_j, avg_j, fig, t)
        
        # plt.show()
        if os.path.exists(f'../results/intersecs/all/') == False:
            os.mkdir(f'../results/intersecs/all/')
        plt.savefig(f'../results/intersecs/all/{name_i}&{name_j}.png')
