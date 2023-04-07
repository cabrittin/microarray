"""
@name:  test_aggrigator.py                      
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import os
import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import DistanceMetric
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import pickle
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.spatial import distance
from collections import defaultdict
from sklearn.cluster import KMeans
from tqdm import tqdm
import scipy.cluster.hierarchy as sch
from mpl_toolkits import mplot3d

from sctool.sc_extension import SCAggregator as SingleCell
import sctool.decomposition as decomp
from sctool import scale
from pycsvparser import read
import toolbox.matrix_properties as mp
from sctool import explore
from toolbox.plots import viz_scatter,add_hover
from toolbox.stats.basic import ecdf
from toolbox.scale import standardize

def load_sc(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    return sc 

def load_lr_pairs(sc):
    _lrpairs = read.into_list('mat/ligand_receptor_pairs.csv',multi_dim=True)
    l,r = zip(*_lrpairs)
    genes = list(set(l+r))
    sc.filter_genes_isin(sc.gene_key,genes)
    sc.index_genes()
    
    lrpairs = []
    for (l,r) in _lrpairs:
        try:
            _ = sc.gene_idx[l]
            _ = sc.gene_idx[r]
            lrpairs.append((l,r))
        except:
            pass
    sc.lrpairs = lrpairs

def lr_aggregator(sc,transpose=False,high_thresh=15,low_thresh=6):
    A1 = format_high_aggregator(sc,thresh=high_thresh)
    a1sum = A1.sum(1)
    a1sum[a1sum==0] = 1
    
    A0 = format_low_aggregator(sc,thresh=low_thresh)
    a0sum = A0.sum(1)
    a0sum[a0sum==0] = 1

    G1 = np.zeros((sc.X.shape[0],len(sc.lrpairs)))
    G0 = np.zeros((sc.X.shape[0],len(sc.lrpairs)))
    for (k,(l,r)) in enumerate(sc.lrpairs):
        if transpose: 
            L = np.sqrt(sc.gene_outer(l,r)).T
        else:
            L = np.sqrt(sc.gene_outer(l,r))
        G1[:,k] = np.divide(np.multiply(A1,L).sum(1),a1sum)
        G0[:,k] = np.divide(np.multiply(A0,L).sum(1),a0sum)
        
    G = np.log10(G1 + 1) - np.log10(G0 + 1)
    G = (G - G.mean()) / G.std()
    return G 

def zscored_lr_aggregator(sc,targets=None,**kwargs):
    G = lr_aggregator(sc,**kwargs) 
    if targets is not None:
        ndx = [sc.nodes.index(t) for t in targets]
        G = G[ndx,:]
    A = format_high_aggregator(sc) 
    lrdx = np.where(A.sum(0)>0)[0]
    sc.lrpairs = [sc.lrpairs[i] for i in lrdx]
    G = G[:,lrdx] 
    G = (G - G.mean()) / G.std()
    return G 

def format_high_aggregator(sc,thresh=15):
    _A = sc.get_aggregator(edge_attr='id') 
    A = np.zeros(_A.shape) 
    A[_A >= thresh ] = 1
    return A 

def format_low_aggregator(sc,thresh=6):
    _A = sc.get_aggregator(edge_attr='id') 
    A = np.zeros(_A.shape) 
    A[_A < thresh] = 1
    return A 

def mean_expression(sc):
    #sc.filter_hvg(num_genes=2000) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    _A = sc.get_aggregator(edge_attr='id') 
    A = np.zeros(_A.shape) 
    A[_A == 18 ] = 1
    E18 = sc.local_mean_aggregate(A)
    
    A = np.zeros(_A.shape) 
    A[_A == 1 ] = 1
    E1 = sc.local_mean_aggregate(A)
    E = E18 - E1
    
    return E
 
def mean_weighted_expression(sc):
    #sc.filter_hvg(num_genes=2000) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    _A = sc.get_aggregator(edge_attr='id') 
    A = sc.get_aggregator(edge_attr='wnorm') 
    A[_A < 18 ] = 0
    asum = A.sum(1)
    asum[asum==0] = 1
    A = A/ asum[:,None]
    E18 = sc.local_mean_aggregate(A)

    A = sc.get_aggregator(edge_attr='wnorm') 
    A[_A > 1 ] = 0
    asum = A.sum(1)
    asum[asum==0] = 1
    A = A/ asum[:,None]
    E1 = sc.local_mean_aggregate(A)
    E = E18 - E1
    
    return E
    
def mean_difference(sc):
    #sc.filter_hvg(num_genes=200) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator(edge_attr='id') 
    A = np.zeros(sc.A.shape) 
    A[sc.A == 18 ] = 1
    E18 = sc.local_diff_aggregate(A=A)
    
    A = np.zeros(sc.A.shape) 
    A[sc.A == 1 ] = 1
    E1 = sc.local_diff_aggregate(A=A)
    
    E = E18 - E1
    return E

def view_adjacency(params):
    sc = load_sc(params)
    tcells = read.into_list(params.target_cells)
    sc.load_aggregator()
    
    tneighs = defaultdict(int)
    top = True
    if top:
        for u in sc.G.nodes():
            if u in tcells: continue
            _tneigh = []
            for v in sc.G.neighbors(u):
                if v not in tcells: continue 
                if sc.G[u][v]['id'] < 16: continue
                _tneigh.append(v)
            tnstr = '-'.join(sorted(_tneigh))
            tneighs[tnstr] += 1
    else:
        for u in tcells:
            _tneigh = []
            for v in sc.G.neighbors(u):
                if v in tcells: continue
                if sc.G[u][v]['id'] < 16: continue
                _tneigh.append(v)
            tnstr = '-'.join(sorted(_tneigh))
            tnstr = u + ':' + tnstr
            tneighs[tnstr] += 1
        u = 'RME_DV' 
        for v in sc.G.neighbors(u):
            print(u,v,sc.G[u][v]['id'])

    for (k,v) in sorted([(k,v) for (k,v) in tneighs.items()],key=lambda x: x[1]):
        print('here',k,v)
            


def run_aggregator(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    
    if params.genes != 'all':
        genes = read.into_list(params.genes)
        sc.filter_genes_isin(sc.gene_key,genes)
    else: 
        sc.filter_hvg(num_genes=2000) 
    
    if params.weighted:
        E = mean_weighted_expression(sc)
    else:  
        E = mean_expression(sc)
    cols = sc.nodes

    wstr = ['','_weighted'][params.weighted]
    dout = f'{params.dout}{params.genes.split(".")[0]}/{params.norm_method}{wstr}/'
    if not os.path.exists(dout): os.makedirs(dout)
    print(f'Writing to {dout}')
    
    if params.target_cells is not None:
        tcells = read.into_list(params.target_cells)
        ndx = [sc.nodes.index(t) for t in tcells]
        E = E[ndx,:].T
        cols = tcells
        X = sc.X[ndx,:].T
    
    ## Standardize E 
    mu = E.mean()
    std = E.std()
    print(std)
    E = (E - mu) / std
       
    #sns.clustermap(E,metric=params.metric,cmap='PRGn',yticklabels=cols)
    #plt.savefig(f'{dout}clustermap.png',dpi=300) 
    
    decomp.tsne(sc,X=E,n_components=2,init='pca',random_state=0,perplexity=30)
    
    rows = sc.genes[sc.gene_key].tolist()
    saved_tsne = {'tsne':sc.D.components,'E':E,
                        'cols':cols,'rows':rows,
                        'X':X}
    
    with open(f'{dout}tsne.pickle', 'wb') as outfile:
        pickle.dump(saved_tsne,outfile,protocol=pickle.HIGHEST_PROTOCOL)
    plt.show()

def plot_tsne(fig,params,key='E',cmap='PRGn'):
    tcells = read.into_list(params.target_cells)
    data =  pickle.load( open( params.fin, "rb" ) )
    
    x1 = data['tsne'][:,0]
    x2 = data['tsne'][:,1]

    grid = AxesGrid(fig, 111,
                nrows_ncols=(4, 4),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )

    for (i,ax) in enumerate(grid):
        ax.set_axis_off()
        if i >= len(tcells): continue
        ndx = data['cols'].index(tcells[i])
        col = data[key][:,ndx]
        if key == 'E': 
            col[col>2] = 2
            col[col<-2] = -2
        im = ax.scatter(x1,x2,s=2,c=col,cmap=cmap)
        ax.text(0.05,0.9,f'{tcells[i]}',transform=ax.transAxes,fontsize=6)
    
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.ax.set_yticks([-2,0,2])
    #cbar.ax.set_yticklabels([-2,0,2]) 
    plt.tight_layout()

def plot_target(params):
    fig = plt.figure(figsize=(8,4.8))
    plot_tsne(fig,params,key='X',cmap='Greys')
    fout = params.fin.replace('.pickle','_targets.png')
    plt.savefig(fout,dpi=300) 
    print(f'Wrote to {fout}') 
    plt.show()        
 
def plot_neighborhood(params):
    fig = plt.figure(figsize=(8,4.8))
    plot_tsne(fig,params,key='E',cmap='PRGn')
    fout = params.fin.replace('.pickle','_neighborhood.png')
    plt.savefig(fout,dpi=300) 
    print(f'Wrote to {fout}') 
    plt.show()        

def lr_analysis(params):
    sc = load_sc(params) 
    load_lr_pairs(sc)

    tcells = read.into_list(params.target_cells)
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    tcells = read.into_list(params.target_cells)
    G = zscored_lr_aggregator(sc,targets=tcells,transpose=params.lr_transpose) 
    
    if params.lr_transpose:
        lrlabels = [f'{r}:{l}' for (l,r) in sc.lrpairs]
    else:
        lrlabels = [f'{l}:{r}' for (l,r) in sc.lrpairs]
    g = sns.clustermap(G,metric='correlation',cmap='PRGn',vmin=-3,vmax=3,
            yticklabels=tcells,xticklabels=lrlabels,figsize=(10,10))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=6)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=6)
    
    """
    #decomp.tsne(sc,X=G.T,n_components=2,init='pca',random_state=0,perplexity=12)
    decomp.pca(sc,X=G.T,n_components=3)
    x1 = sc.D.components[:,0]
    x2 = sc.D.components[:,1]
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(x1,x2,s=20)
    """ 
    plt.show()

def compile_lr_data(params,zscore_thresh = 2):
    tcells = read.into_list(params.target_cells)
    contact_thresh = 18

    ##LR directions ##
    sc = load_sc(params) 
    load_lr_pairs(sc)
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    G = zscored_lr_aggregator(sc,targets=tcells,transpose=False,high_thresh=contact_thresh) 
    gdx = np.where(G.max(0) > zscore_thresh)[0] 
    LR = [sc.lrpairs[g] for g in gdx]

    ##RL directions ##
    sc.restore_data()
    load_lr_pairs(sc)
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    G = zscored_lr_aggregator(sc,targets=tcells,transpose=True,high_thresh=contact_thresh) 
    gdx = np.where(G.max(0) > zscore_thresh)[0] 
    RL = [(sc.lrpairs[g][1],sc.lrpairs[g][0]) for g in gdx]

    lrpairs = LR + RL
    M = len(lrpairs)
    X = np.zeros((1000,M))
    Y = np.zeros(1000)
    
    sc.restore_data()
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.index_cells()
    sc.index_genes()
    idx = 0
    for (kdx,u) in tqdm(enumerate(tcells),desc='Target cells:'):
        udx = sc.cell_idx[u]
        for v in sc.G.neighbors(u):
            if sc.G[u][v]['id'] < contact_thresh: continue
            Y[idx] = kdx
            vdx = sc.cell_idx[v]
            for (jdx,(l,r)) in enumerate(lrpairs):
                ldx = sc.gene_idx[l]
                rdx = sc.gene_idx[r]
                _l = sc.X[udx,ldx]
                _r = sc.X[vdx,rdx]
                X[idx,jdx] = np.sqrt(sc.X[udx,ldx]*sc.X[vdx,rdx]) 
            idx += 1
    X = X[:idx,:]
    Y = Y[:idx]
    print(X.shape,len(LR),len(RL))    
    fout = 'data/sc_aggr/lr_compile.pickle'
    with open(fout, 'wb') as outfile:
        saved_data = {'X':X,'Y':Y,'lrpairs':lrpairs,'target_cells':tcells}
        pickle.dump(saved_data,outfile,protocol=pickle.HIGHEST_PROTOCOL)
    
    ## KMeans
    n_clusters = 3
    kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=10,
                    max_iter=300,random_state=42)
    kmeans.fit(X)
    
    ## TSNe
    decomp.tsne(sc,X=X,n_components=2,init='pca',random_state=0,perplexity=15)
    #decomp.pca(sc,X=X,n_components=3)
    x1 = sc.D.components[:,0]
    x2 = sc.D.components[:,1]
    fig,ax = plt.subplots(1,1,figsize=(2,2))
    cmap = plt.get_cmap('tab10', n_clusters) 
    im = ax.scatter(x1,x2,s=4,c=kmeans.labels_,cmap=cmap,vmin=-0.5,vmax=n_clusters-0.5) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('TSNE 1',fontsize=8)
    ax.set_ylabel('TSNE 2',fontsize=8)
    cax = plt.colorbar(im, ticks=np.arange(0, n_clusters + 1))
    fout = fout.replace('.pickle','_kmeans.svg')
    plt.savefig(fout)

    ## Pioneer contacts across KMeans
    Z = np.zeros((len(tcells),n_clusters))
    for (i,k) in enumerate(kmeans.labels_):
        Z[int(Y[i]),k] += 1

    Z = Z / Z.sum(1)[:,None]
    fig,ax = plt.subplots(1,1,figsize=(1.5,2))
    sns.heatmap(Z,ax=ax,cmap='binary',square=True,vmin=0,vmax=1,
            cbar_kws={"shrink":0.5,"ticks":[0,0.2,0.4,0.6,0.8,1.0]})
    ax.set_xticklabels([0,1,2],fontsize=6)
    ax.set_yticks(np.arange(len(tcells)) + 0.5)
    ax.set_yticklabels(tcells,fontsize=6,rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=4) 
    fout = fout.replace('_kmeans.svg','_kmeans_cell_dist.svg')
    plt.savefig(fout)

   
    ## LR pairs across KMeans
    lrlabels = [f"{l}:{r}" for (l,r) in lrpairs]
    Z = np.zeros((n_clusters,X.shape[1]))
    for (i,k) in enumerate(kmeans.labels_): Z[k,:] += X[i,:]
    Z = Z / Z.sum(1)[:,None] 
    fig,ax = plt.subplots(1,1,figsize=(6,5))
    sns.heatmap(Z,ax=ax,cmap='binary',square=True,vmin=0,vmax=0.3,
            cbar_kws={"orientation":"horizontal",
                "shrink":0.5,"ticks":[0,0.15,0.3]})
    ax.set_yticks([0.5,1.5,2.5]) 
    ax.set_yticklabels([0,1,2],fontsize=6)
    ax.set_xticks(np.arange(len(lrpairs)) + 0.5)
    ax.set_xticklabels(lrlabels,fontsize=6,rotation=45)
    ax.axvline(len(LR),linestyle='--',linewidth=1,color='r') 
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=4) 
    plt.tight_layout() 
    fout = fout.replace('_kmeans_cell_dist.svg','_kmeans_lr_dist.svg')
    plt.savefig(fout)
    plt.show()

def target_vs_neighbors_expression(params):
    sc = SingleCell(params.fin)
    sc.params = params
    sc.filter_hvg(num_genes=2000) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()

    A = sc.get_aggregator(edge_attr='id') 
    W = sc.get_aggregator(edge_attr='wnorm') 
    W1 = np.zeros(W.shape)
    W0 = np.zeros(W.shape)
    W1[A==18] = W[A==18]
    W0[A<6] = W[A<6]
    
    dist1 = defaultdict(list)
    dist0 = defaultdict(list)
    tcells = read.into_list(params.target_cells)
    for t in tcells:
        tdx = sc.nodes.index(t)
        ndx = np.argsort(W1[tdx,:])[::-1][:params.n_neighbors]
        for n in ndx:
            dist1[t].append(distance.cosine(sc.X[tdx,:],sc.X[n,:]))
        ndx = np.argsort(W0[tdx,:])[::-1][:params.n_neighbors]
        for n in ndx:
            dist0[t].append(distance.cosine(sc.X[tdx,:],sc.X[n,:]))
    
    ## Compute baseline
    for (k,v) in dist0.items():
        dist0[k] = [np.mean(v),np.std(v)]
    
    ## compute zscore
    for (k,v) in dist1.items():
        dist1[k] = (np.array(v) - dist0[k][0]) / dist0[k][1]
    
    ## average scores
    mean = np.array([np.mean(dist1[t]) for t in tcells])
    se = np.array([np.std(dist1[t]) for t in tcells]) / np.sqrt(params.n_neighbors)
 
    _col = ['#5fff5a','#9f00ee'] 
    col = [_col[int(mu>=0)] for mu in mean] 
    fig,ax = plt.subplots(1,1,figsize=(3,2))
    x = np.arange(len(tcells)) 
    ax.bar(x,mean,width=0.5,yerr=se,color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(tcells,fontsize=6,rotation=45)
    #ax.set_ylim([0,1]) 
    ax.tick_params(axis='y',labelsize=6)
    #ax.set_ylabel('UMI cosine distance with neighbors',fontsize=8)
    ax.set_ylabel('Neighborhood expression\n distance',fontsize=6)
    plt.tight_layout() 
    if params.dout: plt.savefig(params.dout)
    plt.show()

def aggregate_covariance(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    sc.filter_hvg(num_genes=2000) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.load_aggregator()
    sc.index_cells()
    genes = sc.genes[sc.gene_key].tolist() 
    
    if not os.path.exists(params.dout): os.makedirs(params.dout)
    tcells = read.into_list(params.target_cells)
    for u in tqdm(tcells,desc='Target cells'):
        K,udx = zscore_covariance(sc,u)
        _genes = [genes[dx] for dx in udx]
        if params.dout: 
            fout = f'{params.dout}/{u}_covariance.pickle'
            saved_data = {'K':K,'genes':_genes,'genes_idx':udx}
            with open(fout, 'wb') as outfile:
                pickle.dump(saved_data,outfile,protocol=pickle.HIGHEST_PROTOCOL)
        #sns.heatmap(K,cmap='PRGn')
        y = sch.linkage(K, method='average',optimal_ordering = True)
        im = sns.clustermap(K,row_linkage=y,col_linkage=y,dendrogram_ratio=(.2, .2),
            cbar_pos=(0.75,0.95,0.2,0.03),xticklabels=[],yticklabels=[],
            cmap='PRGn',vmin=-3,vmax=3,figsize=(5,5),
            cbar_kws={"orientation": "horizontal","ticks":[-3,0,3]} )
        im.ax_row_dendrogram.set_visible(False)

        #sns.clustermap(K,metric='cityblock',cmap='PRGn',vmin=-3,vmax=3)
        if params.dout: plt.savefig(f'{params.dout}/{u}_clustermap.png') 
        plt.clf()
        iu = np.triu_indices(K.shape[0])
        kappa = K[iu]
        x,y = ecdf(kappa,reverse=True)
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        ax.plot(x,y)
        ax.set_xlim([-3,3])
        ax.set_ylim([0,1])
        ax.tick_params(axis='x',labelsize=6)
        ax.tick_params(axis='y',labelsize=6)
        ax.set_ylabel('1-ECDF',fontsize=8)
        ax.set_xlabel('Covariance z-score',fontsize=8)
        ax.set_title(f'Cell: {u}',fontsize=8)
        plt.tight_layout()
        if params.dout: plt.savefig(f'{params.dout}/{u}_ecdf.svg') 
        plt.clf()
    #plt.show() 

def zscore_covariance(sc,u): 
    #K = [np.zeros((sc.X.shape[1],sc.X.shape[1])),
    #        np.zeros((sc.X.shape[1],sc.X.shape[1]))]
    W = [np.sum([sc.G[u][v]['wnorm'] 
                    for v in sc.G.neighbors(u) if sc.G[u][v]['id'] < 6]),
         np.sum([sc.G[u][v]['wnorm'] 
                    for v in sc.G.neighbors(u) if sc.G[u][v]['id'] == 18])]
    N = [np.sum([1 for v in sc.G.neighbors(u) if sc.G[u][v]['id'] < 6]),
         np.sum([1 for v in sc.G.neighbors(u) if sc.G[u][v]['id'] == 18])]
    #print(N)
    uvec = sc.get_cell_vector(u)
    udx = np.where(uvec>0)[0]
    uvec = uvec[udx]
    un = len(udx)
    k = np.zeros((un,2))
    k[:,0] = uvec
    K = [np.zeros((un,un)),np.zeros((un,un))]
    keep = [i for i in range(1,6)] + [18]
    for v in sc.G.neighbors(u):
        if sc.G[u][v]['id'] not in keep: continue
        k[:,1] = sc.get_cell_vector(v)[udx]
        cond = int(sc.G[u][v]['id'] == 18)
        w = float(sc.G[u][v]['wnorm']) / W[cond]
        #K[cond]  += w * np.sqrt(np.outer(k[:,0],k[:,1]))
        #K[cond]  += np.cov(k)
        K[cond]  += w*np.cov(k)

    K = K[1] - K[0]
    K = (K - K.mean()) / K.std()
    return K,udx

def build_neighborhood_tree(params):
    tcells = read.into_list(params.target_cells)
    thresh = 2
    G = {}
    for u in tqdm(tcells,desc='Loading data'):
        fin = f'{params.fin}/{u}_covariance.pickle'
        data =  pickle.load( open( fin, "rb" ) )
        data['K'][data['K'] < thresh] = 0
        data['K'][data['K'] > 0] = 1
        G[u] = nx.Graph()
        for (i,j) in zip(*np.where(data['K']>0)):
            G[u].add_edge(data['genes'][i],data['genes'][j])
    
    #for (k,g) in G.items(): print(k,g.number_of_nodes(),g.number_of_edges())
    nodes = []
    for (k,g) in G.items(): nodes += [n for n in g.nodes()]
    nodes = list(set(nodes))
    print('# nodes',len(nodes))
    #for (k,g) in G.items(): G[k].add_nodes_from(nodes)
    #for (k,g) in G.items(): print(k,g.number_of_nodes(),g.number_of_edges())
     
    H = nx.Graph()
    for (k,g) in G.items():
        for (u,v) in g.edges():
            if not H.has_edge(u,v): H.add_edge(u,v,weight=0)
            H[u][v]['weight'] += 1
    
    for n in H.nodes(): H.nodes[n]['count'] = 0
    for (k,g) in G.items():
        for n in g.nodes: H.nodes[n]['count'] += 1

    print(H.number_of_edges())
    freq = np.zeros(len(tcells))
    #for (u,v,w) in H.edges(data='weight'): freq[w-1] += 1
    for n in H.nodes(): freq[H.nodes[n]['count']-1] += 1
    print(freq)    

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    x = np.arange(1,len(tcells)+1)
    ax.bar(x,freq,width=0.25)
    ax.set_xticks(x)
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('# edges',fontsize=8)
    ax.set_xlabel('Frequency',fontsize=8)
    plt.show()

def target_similarity(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    print('shape0',sc.X.shape)
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    sc.filter_hvg(num_genes=2000) 
    sc.X = np.log10(sc.X.toarray() + 1)
    sc.index_cells() 

    sc.index_cells()
    tcells = read.into_list(params.target_cells)
    keep = []
    for u in tcells:
        uvec = sc.get_cell_vector(u)
        keep += np.where(uvec>0)[0].tolist()
        
    keep = list(set(keep))
    U = np.zeros((len(tcells),len(keep)))
    for (i,u) in enumerate(tcells):
        uvec = sc.get_cell_vector(u)
        U[i,:] = uvec[keep]

    #K = np.cov(U)
    K = np.corrcoef(U)
    #K = (K - K.mean()) / K.std()
    
    #sns.set(font_scale=0.8)
    y = sch.linkage(K, method='average',optimal_ordering = True)
    im = sns.clustermap(K,row_linkage=y,col_linkage=y,square=True,
            dendrogram_ratio=(.2, .2), cbar_pos=(0.75,0.95,0.2,0.03),
            xticklabels=[],yticklabels=tcells,
            cmap='PRGn',vmin=-1,vmax=1,figsize=(3,3),
            cbar_kws={"orientation": "horizontal","ticks":[-1,0,1],"shrink":0.3})
    im.ax_row_dendrogram.set_visible(False)
    plt.setp(im.ax_heatmap.get_yticklabels(), fontsize=6)
    cbar = im.ax_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6) 
    
    if params.fout is not None: plt.savefig(params.fout)

    plt.show()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('fin',
                action = 'store',
                help = 'Input file. Config file for aggregator, pickle file for tsne')

   
    parser.add_argument('mode',
                        action = 'store',
                        choices = dir(explore) + [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
   
    parser.add_argument('--filter_level',
                        dest='filter_level',
                        action = 'store',
                        default = -1,
                        required = False,
                        type = int,
                        help = 'Keep cells with filter_level above value')
    
    parser.add_argument('--norm',
                        dest='norm_method',
                        action = 'store',
                        choices = ['norm_global','norm_cell'],
                        default = 'norm_global',
                        required = False,
                        help = 'Specify type of normalization')
    
    parser.add_argument('--weighted',
                    dest='weighted',
                    action = 'store_true',
                    default = False, 
                    required = False,
                    help = 'If flag, use weighted aggregator') 
    
    parser.add_argument('--lr_transpose',
                    dest='lr_transpose',
                    action = 'store_true',
                    default = False, 
                    required = False,
                    help = 'If flag, use transpose of LR aggregator') 
    
    parser.add_argument('--metric',
                        dest='metric',
                        action = 'store',
                        default = 'correlation',
                        required = False,
                        help = 'Metric used for clustermap')
    
    parser.add_argument('--aggregate',
                        dest='aggregator_method',
                        action = 'store',
                        #choices = ['mean_expression','mean_difference'],
                        default = 'mean_expression',
                        required = False,
                        help = 'Aggregator method to be used')

    parser.add_argument('-o',
                        dest='dout',
                        action = 'store',
                        default = 'data/sc_aggr/',
                        required = False,
                        help = 'Output directory')
    
    parser.add_argument('--fout',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to output file')
    
    parser.add_argument('--genes',
                        dest='genes',
                        action = 'store',
                        default = 'all',
                        required = False,
                        help = ('Path to gene list file, if provided '
                            'analysis will be restricted genes in the list'))
    
    parser.add_argument('--target_cells',
                        dest='target_cells',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to list of target cells')

    parser.add_argument('--cell',
                        dest='cell',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Name of a target cell')
    
    parser.add_argument('--n_neighbors',
                        dest='n_neighbors',
                        action = 'store',
                        default = 10,
                        required = False,
                        type = int,
                        help = 'Number of neighbors to use in a clustering algorithm')

    params = parser.parse_args()
    eval(params.mode + '(params)')
