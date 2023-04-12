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


from sctool import explore,util

CONFIG = 'config.ini'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('dataset',
                action = 'store',
                help = 'Dataset alias in config file')
    
    parser.add_argument('--call',
                        action = 'store',
                        dest = 'fn_call',  
                        default = None, 
                        choices = [t for (t,o) in getmembers(explore) if isfunction(o)],
                        required = False, 
                        help = 'Function call')
 
    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')
    
    parser.add_argument('--load_light',
                dest = 'load_light',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If true, loads the meta data but not the count matrix.')
    
    parser.add_argument('--no_display',
                dest = 'no_display',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If true, then do not display plots')
    
    parser.add_argument('--log_scale',
                dest = 'log_scale',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If true, then apply log scale')
    
    parser.add_argument('-o',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Output file')

    args = parser.parse_args()
    
    print(f'Loading dataset {args.dataset}')
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=args.load_light)
   
    while True:
        fn = args.fn_call
        if fn is None: fn = input("Enter to function or exit: ").strip()
        if fn == "switch_log": 
            args.log_scale = not args.log_scale
            continue
        if fn == "exit": break
        
        try:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            getattr(explore,fn)(sc,ax=ax,log_scale=args.log_scale)
            #if not sc.fig_saved and params.fout: plt.savefig(params.fout) 
            if not args.no_display: plt.show()
        except:
            print("Function call not found")
            
        args.fn_call = None

