"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import pandas as pd
from sklearn import manifold


from sctool import explore,util,query,scale
import filters.packer as fltr
import matplotlib.pyplot as plt

import phenograph

cfg = util.checkout('config.ini','packer2019')
sc = util.load_sc(cfg)

fltr.count(sc)
query.qc_mean_var_hvg(sc)
#explore.hvg_mean_var(sc)
#plt.show()
med_libsize = query.median_cell_count(sc)
Xn = scale.normalize_per_cell(sc,counts_per_cell_after=med_libsize,copy=True)
sc.set_count_matrix(Xn)
scale.log1p(sc)
print("Run pca")
query.pca(sc,gene_flag='qc_hvg')
#explore.scree_plot(sc)
#plt.show()
print("Running phenograph")
k = 30
communities, graph, Q = phenograph.cluster(sc.pca.components,k=k)
print(communities)
print(Q)
sc.cells['pheno_clusters'] = pd.Categorical(communities)

perplexity = 15
random_state = 42
fig,ax = plt.subplots(1,1,figsize=(10,10))
tsne = manifold.TSNE(n_components=2,perplexity=perplexity,random_state=random_state)
Z = tsne.fit_transform(sc.pca.components)
ax.scatter(Z[:,0],Z[:,1],c=communities)


plt.show()


