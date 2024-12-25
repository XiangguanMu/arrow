import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# Load the data
data, ts, names, n = dataloader('../datasets/DREAM3/simplified/training_data/InSilicoSize10-Ecoli1-trajectories.tsv')
patch_size = 3
n_poly = 2  # binomial fitting

PTdict_list = []

for i in range(n):
    name = names[i]
    _data = data[:,i]
    _data = [d*1 for d in _data]
    coefficients = []
    deltas = []
    n_ts = len(_data)
    plt.figure(figsize=(10,6))
    plt.xticks(np.arange(n_ts),ts)
    # plt.plot(_data, label='Original series', marker='o')
    for j in range(n_ts-patch_size+1):
        patch = _data[j:j+patch_size]
        coef = np.polyfit(np.arange(patch_size), patch, n_poly)
        coefficients.append(coef)
        d_coef = np.polyder(coef)
        delt = []
        avg = []
        for k in range(patch_size-1):  # k=0,1
            xk = (ts[j+k]+ts[j+k+1])/20  # average of two consecutive points
            avg.append(xk)
            d_x = np.polyval(d_coef, xk-j)
            delt.append(d_x)
            linelen = 0.5
            x_pre = xk-linelen
            x_start = xk
            x_end = xk+linelen
            y_pre = np.polyval(coef,x_start-j)-d_x*linelen
            y_start = np.polyval(coef,x_start-j)
            y_end = np.polyval(coef,x_start-j)+d_x*linelen

            plt.plot([x_pre,x_start,x_end], [y_pre,y_start,y_end], color='green',linewidth=1)            
        

        x_fine = np.linspace(j, j+patch_size-1, 100)  # generates more dots to plot a more smooth curve
        fitted_patch_fine = np.polyval(coef, x_fine-j)  # offset
        plt.plot(x_fine, fitted_patch_fine, linestyle='--')
    plt.title(f'Fitted patches and derivatives at avgs for {name}')
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    # plt.savefig(f'../results/derivatives/{name}.png')
    