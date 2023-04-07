"""
@name: query.py
@description:

Module for plotting qc

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def knee_plot(counts,ax=None,max_counts=None,min_counts=None,**kwargs):
    if max_counts: counts = counts[counts < max_counts]
    if min_counts: counts = counts[counts > min_counts]
    counts = sorted(counts,reverse=True)
    x = np.arange(len(counts))
    ax.plot(x,counts)
    ax.set_yscale('log')
    ax.set_ylabel('Total UMIs',fontsize=8)
    ax.set_xlabel('Ranked UMI',fontsize=8)
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)


def ranked_gene_expr(df,ax=None,**kwargs):
    if not ax:
        height = (df.shape[1] * 0.2) + 1.5
        fig, ax = plt.subplots(figsize=(5, height))

    sns.boxplot(data=df, orient='h', ax=ax, fliersize=1, **kwargs)
    ax.set_xlabel('% of total counts')

    return ax

def violin(data,ax=None,xlabel=None,ylabel=None,title=None,**kwargs):
    if not ax: fig, ax = plt.subplots(1,1,figsize=(3,3))
    
    sns.violinplot(data=data,ax=ax,color='0.6',**kwargs)
    sns.stripplot(data=data,jitter=4,ax=ax,size=1,color='0.3',zorder=1,alpha=0.3,**kwargs)
    ax.tick_params(axis='y', labelsize=6) 
    ax.set_xticks([])
    if xlabel: ax.set_xlabel(xlabel,fontsize=8)
    if ylabel: ax.set_ylabel(ylabel,fontsize=8)
    if title: ax.set_title(title,fontsize=8)

    return ax

def scatter(x,y,ax=None,xlabel=None,ylabel=None,title=None,**kwargs):
    if not ax: fig, ax = plt.subplots(1,1,figsize=(3,3))
    
    ax.scatter(x,y,color='0.6',s=0.8,**kwargs)
    ax.grid()
    ax.set_xlim([0,x.max()])
    ax.set_ylim([0,y.max()])
    ax.tick_params(axis='y', labelsize=6) 
    ax.tick_params(axis='x', labelsize=6) 
    if xlabel: ax.set_xlabel(xlabel,fontsize=8)
    if ylabel: ax.set_ylabel(ylabel,fontsize=8)
    if title: ax.set_title(title,fontsize=8)

    return ax

def hvg_dispersion(_df,dispersion_key='dispersion',ax=None,**kwargs):
    if not ax: fig, ax = plt.subplots(1,1,figsize=(3,3))
    df = _df[['mean',dispersion_key]].copy()
    hvg = np.array(_df['highly_variable_nbatches'].values.tolist())
    #hvg[hvg > 0] = 1
    df['# batches'] = hvg

    sns.scatterplot(ax=ax,data=df,x='mean',y=dispersion_key,hue='# batches',s=1.3)
    ax.set_xlabel('mean gene count',fontsize=8)
    ax.set_ylabel(dispersion_key,fontsize=8)
    ax.tick_params(axis='y', labelsize=6) 
    ax.tick_params(axis='x', labelsize=6) 
    ax.legend(title='# batches',fontsize='6',title_fontsize='6',loc='upper right')
    plt.tight_layout()







