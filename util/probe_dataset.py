"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import numpy as np
import matplotlib.pyplot as plt
import inspect


from sctool.sc_extension import SCFeatureSelect as SingleCell
from sctool import explore


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('config',
                action = 'store',
                help = 'Config file')

   
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

    parser.add_argument('-nz',
                    dest='nz',
                    action = 'store_true',
                    default = False, 
                    required = False,
                    help = 'If flag, skip zeros') 
    
    parser.add_argument('--display_off',
                    dest='display_off',
                    action = 'store_true',
                    default = False, 
                    required = False,
                    help = 'If flag, do not display plot') 


    parser.add_argument('-o',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Output file')

    params = parser.parse_args()
    
    sc = SingleCell(params.config)
    sc.params = params
    
    print('Initial shape:',sc.X.shape)
    if params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
    print('Cell filter:',sc.X.shape)
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    getattr(explore,params.mode)(sc,ax=ax)
    if not sc.fig_saved and params.fout: plt.savefig(params.fout) 
    if not params.display_off: plt.show()

