"""
@name: query.py
@description:

Module for plotting queries

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def ranked_gene_expr(df,ax=None,**kwds):
    if not ax:
        height = (df.shape[1] * 0.2) + 1.5
        fig, ax = plt.subplots(figsize=(5, height))

    sns.boxplot(data=df, orient='h', ax=ax, fliersize=1, **kwds)
    ax.set_xlabel('% of total counts')

    return ax


