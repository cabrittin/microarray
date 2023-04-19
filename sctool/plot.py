"""
@name: plot.py                         
@description:                  
    Highlevel plotting functions

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sctool import query
from toolbox.stats.basic import ecdf

def build_umap(X,**kwargs): 
    import umap.umap_ as umap
    reducer = umap.UMAP(**kwargs)
    reducer.fit(X)
    embedding = reducer.transform(X)
    return embedding,reducer
    #np.save(params.embedding, embedding)
    #pickle.dump(reducer, open(params.transform,'wb'))

def gene_by_label(df,gene,label,ax=None,callback=None,show=True,**kwargs):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    if callback is None:
        sns.violinplot(ax=ax,data=df,x=label,y=gene,**kwargs)
    else:
        sns.violinplot(ax=ax,data=callback(df,gene,label,**kwargs),x=label,y=gene,**kwargs)
    
    if show: plt.show()

def scatter(x,y,ax=None,show=True,**kwargs):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(x,y,s=5,c='k')#,label='High variable genes')
    if show: plt.show()
     
