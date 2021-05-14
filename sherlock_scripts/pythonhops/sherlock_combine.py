#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 01:41:52 2020

Combines files in a directory. 

Required command-line args : folder= , duration= 
Optional command-line args : template= , file_out=

@author: andreypoletaev
"""

# =============================================================================
# %% Imports and constants
# =============================================================================

import pandas as pd

import sys

from glob import glob

from numpy import sqrt, logspace, log10, arange, diff
from numpy import round as npround

# =============================================================================
# %% parse input and combine
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0],x.split('=')[1]) for x in sys.argv[1:] ])

keys = list(options.keys())

# print(options)

assert 'folder' in keys and 'template' in keys, \
    'please pass folder=... [path] and template=... [path with wildcards] as command-line options'

if 'duration' not in keys :
    duration = int(options['template'].split('*')[-1].replace('ps.csv',''))
    print(f'assuming duration {duration} from template {options["template"]}')

# template = f'/*_vacf_{int(options["duration"])}ps.csv' if 'template' not in keys else options['template']
template = options['template']

file_out = f'./vacf_{duration}ps.csv' if 'file_out' not in keys else options['file_out']

print('looking for files that look like this: '+options['folder'] +template) 

## INPUT PARAMETER: computation option
if 'option' not in keys : 
    option = 'standard'
    options['option'] = option
    print('assuming default computation: ensemble averaging')
else : 
    option = options['option']
    print(f'option = {option}')
    
## INPUT PARAMETER: check for 2D constraint
if '2D' in keys :
    dim = 2
elif '2d' in keys : 
    dim = 2
else : dim = 3

output = pd.DataFrame()
counter = 0

found_files = glob(options['folder'] +template)
print(f'Found {len(found_files)} files with pattern {options["folder"] +template}')

for fin in found_files:
    
    ## 'dr' is a debugging tool for r2
    if '_dr.csv' in fin : continue
        
    try: 
        df = pd.read_csv(fin)
        
        # print(df.head())
        if len(df) > 0:
            output = output.append(df, ignore_index=True)
            counter += 1
            if counter % 10 == 0 : print(f'appended data from file #{counter} : {fin}', flush=True)
    except: print(f'could not load / add {fin}')
    
## ensemble-average in almost all cases - but not always the first thing
if option not in ['vanhove', 'dx'] :
    output = output.groupby('time').agg('mean').reset_index()

if option == 'standard' :
    ## automatically calculate the NGP if r2 and r4 are available
    if 'r2' in output.columns and 'r4' in output.columns :
        output['a2'] = 0
        iloc1 = output.time.iloc[1]
        output.loc[iloc1:,'a2'] = (dim/(dim+2)) * output.loc[iloc1:,'r4'] / output.loc[iloc1:,'r2'] **2 -1 
elif option == 'eb' :
    output['eb'] = (output.r4 - output.r2**2) / output.r2**2
    output = output.drop(columns=['r2','r4']) 
elif option == 'vanhove' :
    ## bin the many-start trajectories to get van Hove function
    output = output.set_index('time').stack().reset_index(level=1, drop=True).apply(sqrt).reset_index()
    output.columns = ['time', 'r']
    output['gs'] = output.r
    spacebins = npround(arange(0, output.r.max()+0.03, 0.02),2)
    # print(f'\nmax r = {output.r.max():.3g} simulation boxes')
    output = output.groupby(['time', pd.cut(output.r, spacebins)]).gs.agg('count')
    # print('\nafter groupby:')
    # print(output.head(10))
    output = output.reset_index()
    output.r = output.r.apply(lambda x : x.mid)
    # print('\nfinal form:')
    # print(output.head(10))
elif option == 'dx' :
    ## this is similar to the van Hove function
    output = output.set_index('time').stack().reset_index(level=1, drop=True).reset_index()
    output.columns = ['time', 'dx']
    output['prob'] = output.dx
    spacebins = npround(arange(output.dx.min()-0.02, output.dx.max()+0.03, 0.02),2)
    output = output.groupby(['time', pd.cut(output.dx.copy(deep=True), spacebins)]).prob.agg('count')
    output = output.reset_index()
    output.dx = output.dx.apply(lambda x : x.mid)
elif option == 'burnett' :
    output['x4'] = (output.r4-(dim+2)*(output.r2**2)/dim) 
    output['burnett'] = 0
    
    dx4dt = diff(output.x4,1)/diff(output.time.values,1)
    time_midpoints = output.time.values[1:] - output.time.values[:1]
    output.burnett.iloc[1:-1] = diff(dx4dt,1)/diff(time_midpoints,1) /24
    output = output.drop(columns=['x4'])
                
    
## repair a deleted a2 file that was written directly by lammps
if 'repair' in keys and options['repair'] :
    output.time = (output.time.values * 1000).astype(int)
    output.set_index('time', inplace=True)
    output.to_csv(file_out, sep=' ', )
    
## write file normally
else : output.to_csv(file_out, index=False, float_format='%.7g')












