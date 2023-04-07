"""
@name: ecdf.py
@description:

Module for ECDF plots

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from toolbox import matrix_properties

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs,ys

def plot_ecdf(ax,data,xlabel=None,ylabel='Proportion'):
    xs,ys = ecdf(data) 
    ax.plot(xs,ys,color='k') 
    if xlabel: ax.set_xlabel(xlabel,fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim([0,xs[-1]])
    ax.set_ylim([0,1])
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

def counts_per_cell(sc,ax=None,subsample=100,min_counts=-1,max_counts=1e15,xlabel='Counts per cell'):
    counts = matrix_properties.axis_counts(sc.X,axis=1)
    idx = matrix_properties.counts_in_range(counts,min_counts,max_counts)
    counts = counts[idx]
    if not ax: fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_ecdf(ax,counts[::subsample],xlabel=xlabel)
    return ax

def genes_per_cell(sc,ax=None,subsample=100,min_genes=-1,max_genes=1e15,xlabel='Genes per cell'):
    counts = matrix_properties.axis_elements(sc.X,axis=1)
    idx = matrix_properties.counts_in_range(counts,min_genes,max_genes)
    counts = counts[idx] 
    if not ax: fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_ecdf(ax,counts[::subsample],xlabel=xlabel)
    return ax 

def counts_per_gene(sc,ax=None,subsample=100,min_counts=-1,max_counts=1e15,xlabel='Counts per cell'):
    counts = matrix_properties.axis_counts(sc.X,axis=0)
    idx = matrix_properties.counts_in_range(counts,min_counts,max_counts) 
    counts = counts[idx] 
    if not ax: fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_ecdf(ax,counts[::subsample],xlabel=xlabel)
    return ax

def cells_per_gene(sc,ax=None,subsample=100,min_cells=-1,max_cells=1e15,xlabel='Cells per gene'):
    counts = matrix_properties.axis_elements(sc.X,axis=0)
    idx = matrix_properties.counts_in_range(counts,min_cells,max_cells) 
    counts = counts[idx] 
    if not ax: fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_ecdf(ax,counts[::subsample],xlabel=xlabel)
    return ax
