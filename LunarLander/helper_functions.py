import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

#----------------------------------------------

def read_test_data(net_type, n_runs, names):
    data = []
    for run in range(n_runs):
        file = "../MultipleTesting/%s_Testing/Test_%s_LunarLander_run%d.csv" % (net_type, net_type, run)
        data.append(pd.read_csv(file, names=names))
    return data

#----------------------------------------------

def plot_singleTest(data1, zero_line=[1000,1000], nrows=1, ncols=2, figsize=(12,5), font_size=18, 
                    save_fig=False, fname=["CS"]):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax0, ax1 = axes.flatten()

    plt.setp(ax0.get_xticklabels(), fontsize=font_size)
    plt.setp(ax0.get_yticklabels(), fontsize=font_size)
    ax0.hist(data1["Rewards"], bins=40, color="red")
    ax0.plot([0.0, 0.0],[0 , zero_line[0]],'b-.',linewidth=3)

    plt.setp(ax1.get_xticklabels(), fontsize=font_size)
    plt.setp(ax1.get_yticklabels(), fontsize=font_size)
    ax1.hist(data1["Time"], bins=40, color="red")
    ax1.plot([0.0, 0.0],[0 , zero_line[1]],'b-.',linewidth=3)
    
    if save_fig:
        plt.savefig('fig_%s.png' % (fname[0]), dpi=300, bbox_inches = 'tight')
    plt.show()
    
#----------------------------------------------    

def plot_comparePair(data1, data2, zero_line=[1000,1000], nrows=1, ncols=2, figsize=(12,5), font_size=18, 
                     save_fig=False, fname=["CS","S"]):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax0, ax1 = axes.flatten()

    plt.setp(ax0.get_xticklabels(), fontsize=font_size)
    plt.setp(ax0.get_yticklabels(), fontsize=font_size)
    ax0.hist(data1["Rewards"]-data2["Rewards"], bins=40, color="red")
    ax0.plot([0.0, 0.0],[0 , zero_line[0]],'b-.',linewidth=3)

    plt.setp(ax1.get_xticklabels(), fontsize=font_size)
    plt.setp(ax1.get_yticklabels(), fontsize=font_size)
    ax1.hist(data1["Time"]-data2["Time"], bins=40, color="red")
    ax1.plot([0.0, 0.0],[0 , zero_line[1]],'b-.',linewidth=3)
    
    if save_fig:
        plt.savefig('fig_%s_vs_%s.png' % (fname[0], fname[1]), dpi=300, bbox_inches = 'tight')
    plt.show()
    
#----------------------------------------------

def prepare_f0s_and_f1s(data, n_runs, rows, cols):
    m=rows
    n=cols
    f0, f1 = np.zeros((m,n)), np.zeros((m,n))
    for i in range(n_runs):
        f0[:,i] = np.array(data[i]["Rewards"]).T
        f1[:,i] = np.array(data[i]["Time"]).T
    return f0, f1
    
#----------------------------------------------

def plot_Density(f0_type0, f1_type0, f0_type1, f1_type1, n_runs=1, zero=None, labels=None, save=False, fname=None):
    
    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes.flatten()
    sns.set(font_scale=1.5)

    for run in range(n_runs):
        sns.distplot(f0_type0[:,run]-f0_type1[:,run], hist=False, kde_kws={"shade": True}, label=labels[run], ax=axes[0])
        sns.distplot(f1_type0[:,run]-f1_type1[:,run], hist=False, kde_kws={"shade": True}, label=labels[run], ax=axes[1])
    if zero:
        ax0.plot([0.0, 0.0],[0 , zero[0]],'k-.',linewidth=3)
        ax1.plot([0.0, 0.0],[0 , zero[1]],'k-.',linewidth=3)
    if save:
        plt.savefig(fname, dpi=300, bbox_inches = 'tight')
