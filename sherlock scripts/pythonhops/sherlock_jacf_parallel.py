#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:55:08 2020

@author: andreypoletaev

Inputs:
    file = ... Required
    duration = ... Required
    file_out = ... Required
    com = ... Boolean, default: False. True if center-of-mass velocity.
    batch = ... Batch size for parallel. Default: 20. Not used if not included.
    dims = ... Dimensions. Default: "['x', 'y', 'z']" , interpreted w/ eval().

Assumptions made: 
    time is [picoseconds], timestep is 1 [fs]
    
Parallelization is done for sherlock via schwimmbad and MPI. 
On sherlock, use: ml python/3.6.1 py-schwimmbad
    
"""

# =============================================================================
# %% Imports & constants
# =============================================================================

import sys
from schwimmbad import MPIPool

# from hop_utils import one_vacf, autocorrelation
from numpy import argwhere

import pandas as pd

from datetime import datetime as dt

import itertools

flatten = lambda l: list(itertools.chain.from_iterable(itertools.repeat(x,1) if isinstance(x,str) else x for x in l))

## column names for the cases that the file is a CoM file or a single-atom velocity file
com_col_names = ['timestep', 'x', 'y', 'z', 'vx', 'vy', 'vz']
vel_col_names = ['atom_id', 'time', 'vx', 'vy', 'vz']

## default set of dimensions to compute: all 3 dimensions
dims = ['x', 'y', 'z']

# =============================================================================
# %% Parse inputs 
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0],x.split('=')[1]) for x in sys.argv[1:] ])

assert 'file' in list(options.keys()) and 'duration' in list(options.keys()), \
    'please pass file= ... [path] and duration= ... [psec] as command-line options'

## destination file
assert 'file_out' in list(options.keys()), 'pass an output path, file_out= ...'
fout = options['file_out']
    
col_names = vel_col_names
header = 0

## read the correct options for the file to be loaded
## center-of-mass is the default: skip 2 rows
if ('com' not in list(options.keys())) or (eval(options['com']) == True) :
    col_names = com_col_names
    header = 2

## check dimensions
if 'dims' in list(options.keys()): dims = eval(options['dims'])

# ## read a corrected input file     
try :
    fin = pd.read_csv(options['file'], index_col=False)
    if 'time' not in fin.columns : raise IOError
    print(f'Loaded a corrected file {options["file"]} with cols {fin.columns.values}')
except : 
    print(f'file {options["file"]} is not a corrected one.')
    
    ## read an uncorrected input file 
    try :     
        fin = pd.read_csv(options['file'], sep=' ', skiprows=header, names=col_names, index_col=False)
        ## convert time from [steps] to [ps] if the input file has the former
        fin['time'] = fin.timestep / 1000. ## hard-coded conversion from steps to picoseconds
    except : pass

## remove unnecessary columns and set time as index
fin = fin.set_index('time')[['vx','vy','vz']]
for d in ['x','y','z'] : 
    if d not in dims : fin.drop(f'v{d}', axis=1, inplace=True)
    
## read in batch size if one is passed
batch_size = 5001
if 'batch' in list(options.keys()) : batch_size = eval(options['batch'])

## Read the longest (time) lag to be computed and a list of all lags to run. 
## If the arg duration is longer than the length of simulation, truncate.
## These lags will map to processes. 
max_lag = eval(options['duration'])
try:
    lags = range(argwhere(fin.index.values > max_lag)[0,0])
except: lags = range(len(fin.index.values))
lag_batches = [lags[i:i+batch_size] for i in range(0,len(lags),batch_size)]

## make up a function for mapping single calls to autocorrelation
def one_autocorrelation(tau):
    n = dt.now()
    print(f'starting lag {tau}, time now: {n.strftime("%Y %b %d %H:%M:%S")}')
    cf = dict(zip(sorted(dims),fin.apply(lambda col: col.autocorr(tau))))
    print(f'computed lag {tau}, seconds taken: {(dt.now()-n).total_seconds():.2f}')
    return cf

## make up a function for batch computing autocorrelations
def batch_autocorrelation(taus):
    n = dt.now()
    print(f'starting batch of lags {taus}, time now: {n.strftime("%Y %b %d %H:%M:%S")}')
    cf = [dict(zip(sorted(dims),fin.apply(lambda col: col.autocorr(t)))) for t in taus]
    print(f'computed batch of lags {taus}, seconds taken: {(dt.now()-n).total_seconds():.2f}', flush=True)
    return cf

## shut down all processes except the master one that will map tasks to others
pool = MPIPool()
if not pool.is_master():
    print('one worker on standby')
    pool.wait()
    sys.exit(0)
    
print('MPI master proceeding to map lags to workers...')
print(f'There are {len(lag_batches)} total batches for parallelization')
    
## do the actual parallel computation of the autocorrelation function
print(f'{dt.now().strftime("%Y %b %d %H:%M:%S")}, computing from {options["file"]}')
if 'batch' in list(options.keys()) : acf = flatten(pool.map(batch_autocorrelation, lag_batches))
else : acf = pool.map(one_autocorrelation, lags)
print(f'done with parallel computation, {dt.now().strftime("%Y %b %d %H:%M:%S")}')

## convert to dataframe
acf = pd.DataFrame(acf, index=fin.index.values[:len(lags)]).reset_index().rename(columns={'index':'time'})

## save output
acf.to_csv(fout, index=False, float_format='%.7g')
print(f'computed {options["file"]} and saved to {options["file_out"]}')

## close the MPI pool
pool.close()














