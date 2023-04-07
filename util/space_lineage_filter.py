"""
@name:
@description:


@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from tqdm import tqdm

from cebraindev.cetorch.cedata import CeData
from toolbox import plots


from pycsvparser import write

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

def plot_unit_circle(theta,**kwargs):
    x = np.cos(theta)
    y = np.sin(theta)
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(x,y,**kwargs)
    plt.show()

def zscore_binarize(X,zscore=0):
    eps = 1e-5 
    X = (X - X.mean(0)) / (X.std(0) + eps)
    return np.where(X>=zscore,1,0)

def split_domain(X,pos_dom,neg_dom=None):
    xpos = X[:,pos_dom].sum(1)
    if neg_dom is not None:
        xneg = X[:,neg_dom].sum(1) 
    else:
        xneg = 1 - xpos
    return xpos,xneg

def parse_domains(domains,num_levels=None):
    if num_levels is None: num_levels = len(domains)-1
    doms = []
    for i in range(1,num_levels+1):
        doms += [list(c) for c in combinations(domains,i)]
    return doms

def adjusted_formula(Y,X,Z,pz=None):
    eps = 1e-5
    if pz is None: pz = Z.sum(0) / float(Z.sum())
    psum = 0
    for i in range(Z.shape[1]):
        xz = X*Z[:,i]
        yxz = Y*xz
        pjoint = yxz.sum() / (xz.sum() + eps)
        psum += pjoint * pz[i]
    return psum

def compute_ace(domain,X,Y,Z):
    pz = Z.sum(0) / float(Z.sum())
    xpos,xneg = split_domain(X,domain) 
    ace = np.zeros(Y.shape[1])  
    for i in range(Y.shape[1]):
        pos = adjusted_formula(Y[:,i],xpos,Z,pz=pz)
        neg = adjusted_formula(Y[:,i],xneg,Z,pz=pz)
        ace[i] = pos-neg  
    return ace

def load_data(D):
    theta_s = parameterize_theta(D.target)
    shift_theta(theta_s,0.35*np.pi)
    label_s = encode_theta(theta_s)
    ohs = encode_onehot(label_s)
    #plot_unit_circle(theta_s,s=20,c=label_s,cmap='Set1')    
    
    theta_l = parameterize_theta(D.lineage)
    label_l = encode_theta(theta_l)
    ohl = encode_onehot(label_l)
    
    D.meta['theta_s'] = label_s-1
    D.meta['theta_l'] = label_l-1
    D.save_data()

    labels = list(zip(label_l,label_s))
    return labels,ohs,ohl

def filter(params,ace_thresh=0.1):
    D = CeData(params.data,lineage=True)
    labels,ohs,ohl = load_data(D)
    
    #np.random.shuffle(ohs)
    #np.random.shuffle(ohl)

    Y = zscore_binarize(D.input,zscore=0)
    _doms = [i for i in range(ohs.shape[1])]
    
    num_levels = 1
    doms = parse_domains(_doms,num_levels=num_levels)
    
    print('Space classes',ohs.sum(0))
    print('Lineage classes',ohl.sum(0))
    A = np.zeros((Y.shape[1],len(doms)))      
    for (i,d) in tqdm(enumerate(doms),total=len(doms),desc='Domains'):
        A[:,i] = compute_ace(d,ohs,Y,ohl)
    
    print(A)
    A[A<=ace_thresh] = 0
    
    """
    #Following written to check if ACE is robust to shuffling
    #In general, find that shuffle lead to ACE=0 for all genes
    #Not sure if worth following up on
    M = np.zeros(A.shape,dtype=int)
    M[A>0] = 1
    Ar = np.zeros(A.shape)
    iters = 100
    for j in tqdm(range(iters),desc='Shuffle'):
        np.random.shuffle(ohs)
        np.random.shuffle(ohl)
        for (i,d) in enumerate(doms):
            Ar[:,i] += compute_ace(d,ohs,Y,ohl)
    
    Ar /= iters
    Ar = Ar * M
    print(Ar)
    fig,ax = plt.subplots(1,1,figsize=(2.5,5))
    ax.imshow(Ar,vmin=0,vmax=0.5,cmap='plasma',interpolation='nearest',aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    """

    fig,ax = plt.subplots(1,1,figsize=(2.5,5))
    ax.imshow(A,vmin=0,vmax=0.5,cmap='plasma',interpolation='nearest',aspect='auto') 
    ax.set_xticks([])
    ax.set_yticks([])
    if params.fout: 
        plt.savefig(params.fout)
        print(f'Wrote to {params.fout}')
    

    genes = D.genes['gene_name'].tolist()
    A[A>0] = 1
    zsum = A.sum(1)
    zsum[zsum>0] = 1
    print(A.sum(0))
    print(zsum.sum())
    
    f_genes = [genes[i] for i in np.where(zsum > 0)[0]]
    
    if params.f_genes is not None:
        write.from_list(params.f_genes,f_genes)
        print(f'Wrote to {params.f_genes}')

    plt.show()

def deprecated_level_wrapper(A):
    Z = np.zeros(A.shape)
    for dom_target in [0,1,2,3]:
        levels = setup_levels(num_levels,doms,dom_target) 
        Lvals = level_up(levels,A,ace_thresh=0.1)
        idx = num_levels*dom_target + np.arange(num_levels)
        Z[:,idx] = Lvals
 
def filter_test(params):
    #Generate gene ids
    gids = generate_gene_list(4,4,3) 
    
    #Generate cell ids
    D = CeData(params.data,lineage=True)
    cids,ohs,ohl = load_data(D)
    
    #Build expression matrix
    Y = np.zeros((len(cids),len(gids)))
    for (i,(alpha,beta)) in enumerate(cids):
        for (j,(gamma,etta)) in enumerate(gids):
            Y[i,j] = synthesize_gene(alpha,beta,gamma,etta)
    
    Y[Y>0] = 1
    genes = D.genes['gene_name'].tolist()
    max_genes = len(genes)
    _doms = [i for i in range(ohs.shape[1])]
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(Y)#,vmin=0,vmax=1.5,cmap='plasma')
    ax.set_xticks([])
    ax.set_yticks([])
    if params.fout is not None: 
        fout = params.fout.replace('.png','._syth.png')
        plt.savefig(fout)
        print(f'Wrote to {fout}')

    #plt.show()
    
    num_levels = 1
    doms = parse_domains(_doms,num_levels=num_levels)
    
    A = np.zeros((Y.shape[1],len(doms)))      
    for (i,d) in tqdm(enumerate(doms),total=len(doms),desc='Domains'):
        A[:,i] = compute_ace(d,ohs,Y,ohl)
    
    Z = np.zeros((Y.shape[1],4*num_levels))
    for dom_target in [0,1,2,3]:
        levels = setup_levels(num_levels,doms,dom_target) 
        Lvals = level_up(levels,A,ace_thresh=0.1)
        idx = num_levels*dom_target + np.arange(num_levels)
        Z[:,idx] = Lvals
    
    fig,ax = plt.subplots(1,1,figsize=(2.5,5))
    ax.imshow(Z,vmin=0,vmax=0.5,cmap='plasma',interpolation='nearest',aspect='auto') 
    ax.set_xticks([])
    ax.set_yticks([])
    if params.fout is not None: 
        fout = params.fout.replace('.png','._filter.png')
        plt.savefig(fout)
        print(f'Wrote to {fout}')

   
    plt.show()

def generate_gene_list(num_sd,num_ld,num_grp):
    gids = []
    for i in range(num_ld):
        for j in range(num_sd):
            for k in range(num_grp):
                gids.append((i+1,-1))
    for i in range(num_ld):
        for j in range(num_sd):
            for k in range(num_grp):
                gids.append((i+1,j+1))
    return gids
    
def synthesize_gene(alpha,beta,gamma,etta,mu=1,std=0.1):
    val = 0
    c1 = (alpha == gamma)
    c2 = (etta == -1)
    c3 = (beta == etta)
    if c1 and (c2 or c3):
        val = np.random.normal(loc=mu,scale=std)
    return val

def setup_levels(num_levels,doms,dom_target):
    levels = [[] for i in range(num_levels)]
    for (i,dm) in enumerate(doms):
        if dom_target not in dm: continue
        levels[len(dm)-1].append(i)
    return levels[::-1]
 
def level_up(levels,A,ace_thresh=0.1):
    Ldx = np.zeros([A.shape[0],len(levels)],dtype=int)
    Lvals = np.zeros([A.shape[0],len(levels)])
    for (j,l) in enumerate(levels):
        Lvals[:,j] = A[:,l].mean(1)
    
    ineg = np.where(Lvals >= ace_thresh )
    Ldx[ineg] = 1
    Lvals = Lvals*Ldx

    for i in range(Ldx.shape[1]): Ldx[:,i] = np.prod(Ldx[:,i:],axis=1)
    lsum = Ldx.sum(1)
    lmax = lsum.max() 
    for i in range(Ldx.shape[1]):
        idx = np.where(lsum > lmax)[0]
        Ldx[np.where(lsum > lmax)[0],i] = 0
        lmax -= 1
   
    return Ldx * Lvals


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('data',
                        action = 'store',
                        help = 'Path to npz datafile')
    
    parser.add_argument('mode',
                        action = 'store',
                        help = 'Mode to run')
    
    parser.add_argument('-o',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to outputfile')
    
    parser.add_argument('--f_genes',
                        dest='f_genes',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Output file for filtered genes')

    params = parser.parse_args()
    
    eval(params.mode + '(params)')
