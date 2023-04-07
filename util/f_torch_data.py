"""
@name: f_torch_data.py
@description:
    Format data in TorchData format for downstream analysis. 

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import networkx as nx
from scipy.spatial import distance
from random import shuffle
from tqdm import tqdm

from pycsvparser import read
from sctool.sc_extension import SCAggregator as SingleCell
from sctool import scale
from toolbox.ml.dr import classic_mds as mds
from toolbox.stats.basic import ecdf

SD_COLORS = ['#9301E7','#E7A401','#5E7FF1','#FC0000','#1FFF00','#9b9b9b']
CLADE_COLORS = {'Unc':'#a1a1a1', '-1':'#ff5fd1', 'ABpxa':'#5fccff', 'ABaxppp':'#00b21b', 'ABpxp':'#ffce65', 'ABax':'#ca005f'}

def load_sc(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    
    if params.genes is not None:
        genes = read.into_list(params.genes)
        sc.filter_genes_isin(sc.gene_key,genes)
    else:
        sc.filter_hvg()
    
    print('gene filter',sc.X.shape)
    sc.X = np.log10(sc.X.toarray() + 1) 
    sc.load_aggregator()
    sc.filter_graph_edges(weight='id',min_weight=16)
    sc.load_lineage()
    sc.index_cells() 
    return sc 

def save_data_simple(params,X,Y,L,edge_list,Meta,Gmeta,target_cells=None):
    np.savez_compressed(params.fout,X=X,Y=Y,L=L)#,edge_list=edge_list)
    add_sd_color(Meta)
    add_clade_color(Meta)
    fout = params.fout.replace('.npz','_meta.csv')
    Meta.to_csv(fout,index=False)
    gout = params.fout.replace('.npz','_genes.csv')
    Gmeta.to_csv(gout,index=False)
    print(f'Wrote to {params.fout}')
    print(f'Wrote to {fout}')
    print(f'Wrote to {gout}')
        

def torch_target(params):
    sc = load_sc(params)
    tcells = read.into_list(params.target_cells)
    nodes = sc.cells[sc.cell_key].tolist()
    sc.index_cells()
    nsize = dict([d for d in sc.G.degree(weight='wnorm')])
    
    Z = np.zeros((len(nodes),len(tcells)))
    for i,u in enumerate(nodes):
        for j,v in enumerate(tcells):
            #Z[i,j] = int(G.has_edge(u,v))
            if not sc.G.has_edge(u,v): continue
            #print(G[u][v]['weight']/nsize[u])
            if nsize[u] == 0: print(u)
            #Z[i,j] = sc.G[u][v]['wnorm'] / nsize[v]
            Z[i,j] = sc.G[u][v]['wnorm']
    
    Z = Z / (Z.sum(1) + 1e-5)[:,None]

    edge_list = np.zeros(1)
    #print(sc.L.edges())
    L = nx.to_numpy_array(sc.L,nodelist=nodes,weight='weight')
    #L = nx.laplacian_matrix(sc.L,nodelist=nodes,weight='weight') 
    #L = L.toarray() 
    L = L / L.max()
    save_data_simple(params,sc.X,Z,L,edge_list,sc.cells,sc.genes)

def torch_full(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    print('init',sc.X.shape)
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    
    sc.filter_cells_gt('space_knn_labels',-1)
    print('filter',sc.X.shape)
 
    sc.filter_hvg(num_genes=2000)
    print('gene filter',sc.X.shape)
    sc.X = np.log10(sc.X.toarray() + 1) 
    
    
    edge_list = np.zeros(1)
    L = np.zeros(1)
    Z = np.array(sc.cells['space_knn_labels'].tolist())

    save_data_simple(params,sc.X,Z,L,edge_list,sc.cells,sc.genes)

def torch_gdrde(params):
    sc = SingleCell(params.fin)
    sc.params = params

    print('init',sc.X.shape)
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
     
    #sc.filter_cells_gt('space_knn_labels',-1)
    #print('filter',sc.X.shape)
    sc.load_aggregator()
    
    ##Need to check why RIP is getting dropped
    ## The number of reproducible contacts is low
    #for v in sc.G.neighbors('RIP'):
    #    print(v,sc.G['RIP'][v]['id'])
    #print('00',sorted(sc.cells[sc.cell_key].tolist())) 
    sc.load_lineage()
    
    sc.filter_graph_edges(weight='id',min_weight=18)
    print('11',sc.G.number_of_edges()) 
    
    """
    sc.filter_graph_edges(weight='wnorm',min_weight=0.0005)
    weight = [w for (u,v,w) in sc.G.edges(data='wnorm')]
    x,y = ecdf(weight,reverse=True)
    plt.plot(x,y)
    plt.show()
    """

    nodes = sorted(sc.G.nodes())
    A = nx.to_numpy_array(sc.G,nodelist=nodes,weight=None)
    A = np.dot((np.eye(len(nodes)) + A),A)
    deg = np.diagonal(A).copy()
    np.fill_diagonal(A,0)
    
    L = nx.Graph()
    for (u,v,w) in sc.L.edges(data='weight'):
        if u not in nodes: continue
        if v not in nodes: continue
        if w > 8: continue
        L.add_edge(u,v)
    
    sc_cells = set(sc.cells[sc.cell_key].tolist())
    rm_nodes = set(nodes) - sc_cells
    print('Removing nodes not in expression table')
    print(rm_nodes)
    nodes = sorted(list(set(nodes) - set(rm_nodes)))
    
    rm_nodes = ['ASJ']
    print('Manually removing these nodes due to lack of eventual space neighbors')
    print(rm_nodes)
    nodes = sorted(list(set(nodes) - set(rm_nodes)))

    rm_nodes = set(nodes) - set(L.nodes())
    print('Removing nodes not in lineage')
    print(rm_nodes)
    nodes = sorted(list(set(nodes) - set(rm_nodes)))
    node_map = dict([(n,i) for (i,n) in enumerate(nodes)])
    
    ##Both L and A should have the same row ordering
    L = nx.to_numpy_array(L,nodelist=nodes,weight=None)
    A = nx.to_numpy_array(sc.G,nodelist=nodes,weight=None)
    A = np.dot((np.eye(len(nodes)) + A),A)
    deg = np.diagonal(A).copy()
    np.fill_diagonal(A,0)
    print('num_edges',len(np.where(A>0)[0])*0.5,A.shape[0]*(A.shape[0]-1)) 
    rm_nodes = set(nodes) - set(sc.cells[sc.cell_key].tolist())
    print('Removing nodes not in adjacency')
    print(sorted(rm_nodes))
    sc.filter_cells_isin(sc.cell_key,nodes)
    sc_cells =  sc.cells[sc.cell_key].tolist()
    #sc.filter_hvg(num_genes=2000)
    print('gene filter',sc.X.shape)
    sc.X = np.log10(sc.X.toarray() + 1) 
    
    A[A<2] = 0 
    H = nx.from_numpy_array(A)
    weight = [w for (u,v,w) in H.edges(data='weight')]
    x,y = ecdf(weight,reverse=True)
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(x,y)
    

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    nx.draw_networkx(H,ax=ax,font_size=6,node_size=50)
    plt.show()

    print('00',set(nodes) - set(sc_cells))
    graph_map = [node_map[c] for c in sc_cells]
    #print(graph_map)
    sc.cells['graph_map'] = graph_map
    
    sc.cells['class_labels'] = sc.cells[sc.cell_key].tolist()

    save_data_simple(params,sc.X,A,L,[],sc.cells,sc.genes) 

    
 
def _torch_distance(func):
    def inner(*args):
        params = args[0]
        sc = load_sc(params)
        tcells = read.into_list(params.target_cells)
        
        """
        shuffle(nodes)
        rdx = [sc.cell_idx[n] for n in nodes]
        sc.reorder_rows(rdx)
        nodes = sc.cells[sc.cell_key].tolist()
        """
        nodes = sc.cells[sc.cell_key].tolist()
        sc.index_cells()
        nsize = dict([d for d in sc.G.degree(weight='weight')])

        Z = np.zeros((len(nodes),len(tcells)))
        for i,u in enumerate(nodes):
            for j,v in enumerate(tcells):
                #Z[i,j] = int(G.has_edge(u,v))
                if not sc.G.has_edge(u,v): continue
                #print(G[u][v]['weight']/nsize[u])
                if nsize[u] == 0: print(u)
                Z[i,j] = sc.G[u][v]['weight'] / nsize[u]
        edge_list = []  
        for (u,v) in sc.G.edges():
            if (u not in tcells) and (v not in tcells): continue
            edge_list.append([sc.cell_idx[u],sc.cell_idx[v]])
        edge_list = np.array(edge_list).astype(int)
        
        #D = distance.cdist(Z,Z,'jaccard')
        D = distance.cdist(Z,Z,'cosine') 
        D = np.nan_to_num(D,nan=1)
        np.fill_diagonal(D,0)
        X = sc.X.copy()
        #X = X/X.max()
        print(sorted(sc.L.nodes()))
        L = nx.to_numpy_array(sc.L,nodelist=nodes,weight='weight')
        L = L / L.max()

        D,L,X = func(D,L,X) 
        #D,_ = mds(D,n_components=5)
        #D = D - D.min()
        #D = D / abs(D).max()
        
        save_data_simple(params,X,D,L,edge_list,sc.cells,sc.genes)
        return None
    return inner

@_torch_distance
def torch_distance(*args):
    return args[0],args[1],args[2]

@_torch_distance
def torch_mds(*args):
    D,_ = mds(args[0],n_components=2)
    D = D / abs(D).max()
    L,_ = mds(args[1],n_components=2)
    L = L / abs(L).max()
    return D,L,args[2] 

@_torch_distance
def torch_theta(*args):
    D,_ = mds(args[0],n_components=2)
    D = D / abs(D).max()
    L,_ = mds(args[1],n_components=2)
    L = L / abs(L).max()
    
    theta_s = parameterize_theta(D)
    shift_theta(theta_s,0.35*np.pi)
    labels = encode_theta(theta_s)
    #D = encode_onehot(labels)
    D = np.array(labels)
    theta_l = parameterize_theta(L)
    shift_theta(theta_l,np.pi)
    labels = encode_theta(theta_l)
    L = encode_onehot(labels)

    return D,L,args[2]


@_torch_distance
def torch_cov(*args):
    D,_ = mds(args[0],n_components=2)
    D = D / abs(D).max()
    L,_ = mds(args[1],n_components=2)
    L = L / abs(L).max()
    
    theta_s = parameterize_theta(D)
    shift_theta(theta_s,0.35*np.pi)
    labels = encode_theta(theta_s)
    
    X = args[2]
    #X = X - X.mean(1)[:,None]
    #X = X / X.max(0)
    
    fig,_ax = plt.subplots(2,2,figsize=(16,16))
    for (i,ax) in enumerate(_ax.flatten()): 
        l = i + 1
        x = X[np.where(labels==l)[0],:].T
        cov = np.cov(x)
        cov = cov / abs(cov).max()
        print(cov.min(),cov.max())
        ax.imshow(cov,vmin=-0.05,vmax=0.05,cmap='PRGn')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


    """
    iu = np.triu_indices(X.shape[1],k=1)
    ksize = X.shape[0]*(X.shape[0] - 1) // 2 
    Xo = np.zeros((ksize,len(iu[0])))
    k = 0
    for i in tqdm(range(X.shape[0]),desc='Covariance'):
        for j in range(i+1,X.shape[0]):
            xo = np.outer(X[i,:],X[j,:])
            Xo[k] = xo[iu]
            k += 1
    """

def parameterize_theta(Y):
    rnorm = np.linalg.norm(Y,axis=1)
    Z = Y / rnorm[:,None]
    x,y = (Z[:,0],Z[:,1])
    #r = np.sqrt(x**2 + y**2)
    #theta = np.arccos(x / r)
    theta = np.arccos(x)
    theta[y<0] = -theta[y<0]
    return theta 

def shift_theta(theta,shift=0):
    theta += shift
    mod = np.where(theta > np.pi)
    theta[mod] = theta[mod] - 2*np.pi

def encode_theta(theta,bounds=[-0.5*np.pi,0,0.5*np.pi]):
    cur_id = len(bounds) + 1
    enc_theta = np.ones(theta.shape,dtype=int)*cur_id
    bounds = sorted(bounds,reverse=True)
    for (j,b) in enumerate(bounds): 
        cur_id -= 1
        enc_theta[np.where(theta < b)] = cur_id
    #plot_unit_circle(theta,s=20,c=enc_theta,cmap='Set1')
    return enc_theta

def encode_onehot(labels):
    udx = sorted(np.unique(labels))
    one_hot = np.zeros((len(labels),len(udx)),dtype=int)
    for (i,u) in enumerate(udx):
        one_hot[np.where(labels==u)[0],i] = 1
    return one_hot

def multilabel(parms):
    sc = load_sc(params)
    tcells = read.into_list(params.target_cells)

    nodes = sc.cells[sc.cell_key].tolist()
    sc.index_cells()
    nsize = dict([d for d in sc.G.degree(weight='weight')])

    Z = np.zeros((len(nodes),len(tcells)))
    for i,u in enumerate(nodes):
        for j,v in enumerate(tcells):
            if not sc.G.has_edge(u,v): continue
            #print(G[u][v]['weight']/nsize[u])
            if nsize[u] == 0: print(u)
            #if sc.G[u][v]['weight']/nsize[u] < 0.01: continue
            Z[i,j] = 1 # sc.G[u][v]['weight'] / nsize[v]
    
    edge_list = []  
    for (u,v) in sc.G.edges():
        if (u not in tcells) and (v not in tcells): continue
        edge_list.append([sc.cell_idx[u],sc.cell_idx[v]])
    edge_list = np.array(edge_list).astype(int)

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(Z,vmin=0,vmax=1,cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    L = nx.to_numpy_array(sc.L,nodelist=nodes,weight='weight')
    L = L / L.max()

    save_data_simple(params,sc.X,Z,L,edge_list,sc.cells,sc.genes)
    

def add_sd_color(meta):
    sd = meta['spatial_domain'].tolist()
    color = [SD_COLORS[s] for s in sd]
    meta['sd_color'] = color

def add_clade_color(meta):
    clade = meta['clade'].tolist()
    color = [CLADE_COLORS[c] for c in clade]
    meta['clade_color'] = color

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('fin',
                action = 'store',
                help = 'Input file. Config file for aggregator, pickle file for tsne')
    
    parser.add_argument('mode',
                        action = 'store',
                        help = 'Function call')
   
    parser.add_argument('--filter_level',
                        dest='filter_level',
                        action = 'store',
                        default = -1,
                        required = False,
                        type = int,
                        help = 'Keep cells with filter_level above value')
    
    parser.add_argument('-o','--output',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to output file')
    
    parser.add_argument('--genes',
                        dest='genes',
                        action = 'store',
                        default = None,
                        required = False,
                        help = ('Path to gene list file, if provided '
                            'analysis will be restricted genes in the list'))
    
    parser.add_argument('--target_cells',
                        dest='target_cells',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to list of target cells')

    params = parser.parse_args()
    eval(params.mode + '(params)')
