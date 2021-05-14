#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 03:29:51 2020

Small script to apply transformations to a trajectory using multiple starts.
Most obvious uses: r2 and r4

Command-line options:
    required file=... , starts=... , duration= ... , file_out= ... 
    optional option= ... (default 'multistart_r2r4') for the kind of calculation to do
    optional Lx=..., Ly=..., Lz=... (defaults are 1) for simulation box dimensions
    optional cell_file=... (no default) for where to read cell dimensions
    optional split=... , number of sub-trajectories into which to split input 
    (using split will generate multiple files)
    
    optional func= ... (function with 1 parameter, default: x**2) not implemented

@author: andreypoletaev
"""

from datetime import datetime as dt
t1 = dt.now()

from sys import argv
from hop_utils import multistart_r2r4, multiduration_r2r4

from pandas import read_csv

from numpy import eye, arange, logspace, log10, linspace, loadtxt, floor, isclose, array
from numpy import round as npround

# assumed columns in the trajectory data file
data_cols = ['atom','time', 'x', 'y', 'z']

# keys for cell dimensions - less useful than reading a cell from a file
box_dimension_keys = ['Lx', 'Ly', 'Lz']

# defaults: simulation box, assumed orthogonal; bins for log time, 2D flag
cell = eye(3)
timebins = None
twod = False

# =============================================================================
# %% read options
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0], x.split('=')[1]) for x in argv[1:] ])

keys = list(options.keys())

assert 'file' in keys and 'duration' in keys and 'starts' in keys, \
    'pass file=... [path] , duration=... [psec] , starts=... [iterable] as command-line options'
    
assert 'file_out' in keys, 'pass file_out=... [path] as a command-line option'

## read the option, or else apply the default one
try: option = options['option']
except :
    option = 'multistart_r2r4'
    options['option'] = option
    print('no option provided, assuming default computation: multi-start r2 & r4')

# print(f'starting {options["file"]} at {t1.strftime("%y%m%d %H:%M:%S")}')

## read duration
duration = eval(options['duration'])

## read starts
starts = eval(options['starts'])
if type(starts) is range : starts = array(starts)
starts -= min(starts)

## check for 2D constraint
if '2D' in keys :
    twod = eval(options['2D'])
elif '2d' in keys : 
    twod = eval(options['2d'])

## split trajectories into multiple ones
try: 
    split = int(eval(options['split']))
    print(f'splitting trajectory into {split} parts.')
except: split = 0

if 'com' in keys: com = eval(options['com'])
else : com = False

## load file; assumes CoM file has been corrected
if com: traj_df = read_csv(options['file'], sep=',').set_index('time')
else: traj_df = read_csv(options['file'], sep = ' ', names=data_cols, header=None).drop('atom',axis=1).set_index('time')
traj_df.index -= min(traj_df.index)
ts = traj_df.index[1] - traj_df.index[0]
prec = -int(floor(log10(ts)))
while not isclose(ts % 10**-prec, 0) : prec += 1
# print(f'precision: {prec}')
print(f'First time point: {traj_df.index.values[0]}, {prec} decimal places')

# dtt = (dt.now()-t1).total_seconds()
# print(f'loaded : {options["file"]}, time taken: {dtt:.2f} sec')

## read time bins
## subsample a bit for 'burnett'
if 'timebins' in keys : 
    if options['timebins'] == 'default' :
        timebins = [0] + list(arange(ts,1.0,ts)) + list(arange(1.0,3.0,ts*2)) \
                       + list(npround(logspace(log10(3.0),log10(duration),500)/ts)*ts)
    elif options['timebins'] == 'eb' :
        timebins = list(npround(linspace(duration, traj_df.index.max()*0.9, 10000)))
        # print(f'time bins: {timebins}')
    elif options['timebins'] == 'burnett' :
        timebins = None
        # traj_df = traj_df[::2]
        # print('Timebins are "burnett", use every 2nd point, re-do precision')
        # traj_df.index -= traj_df.index.values[0]
        ts = traj_df.index[1] - traj_df.index[0]
        prec = -int(floor(log10(ts)))
        while not isclose(ts % 10**-prec, 0) : prec += 1
    else : 
        try : timebins = eval(options['timebins'])
        except : 
            print(f'Could not interpret the passed time bins: {options["timebins"]}')
            timebins = None
else : print('No time bins provided, using None')

if option == 'vanhove' and timebins == None:
    if 'long' in keys and eval(options['long']) == True : 
        # timebins = [0] + list(arange(0.101,3.0,0.1)) + list(npround(logspace(log10(3.01),log10(duration+0.01),150),3))
        timebins = [0] + list(arange(ts,1.0,ts)) + list(arange(1.0,3.0,ts*2)) \
                       + list(npround(logspace(log10(3.0),log10(duration),150)/ts)*ts)
    else : 
        # timebins = [0] + list(arange(0.101,3.0,0.1)) + list(npround(logspace(log10(3.01),log10(duration+0.01),120),3))
        timebins = [0] + list(arange(ts,1.0,ts)) + list(arange(1.0,3.0,ts*2)) \
                       + list(npround(logspace(log10(3.0),log10(duration),150)/ts)*ts)
                       
if timebins is not None : 
    timebins = npround(timebins, prec)
    
## scale by the dimensions of the cell if Lx, Ly, Lz are given as inputs
## or read a cell from a file
if all([xyz in keys for xyz in box_dimension_keys]):
    for i, dim in zip([0,1,2],['x', 'y', 'z']) :
        cell[i,i] *= eval(options[f'L{dim}'])
elif 'cell_file' in keys :
    try: 
        cell2 = loadtxt(options['cell_file'])
        if cell2.shape == (3,3) : cell = cell2
    except : 
        print(f'could not load cell from {options["cell_file"]}')
else : 
    print('No cell provided, proceeding with np.eye(3)')
    
## do the calculations
if option == 'multistart_r2r4' :
    if not split:
        out = multistart_r2r4(traj_df, duration, starts, cell, timebins, twod=twod)
    else : 
        outs = list()
        segment_length = len(traj_df) // split
        print(f'Splitting into {split} segments, each {segment_length} points.')
        for i in range(split) :
            traj_df_i = traj_df.iloc[segment_length*i : segment_length * (i+1), :]
            if i > 0 : traj_df_i.index = npround(traj_df_i.index.values - max( min(starts), min(traj_df_i.index.values)), prec)
            outs.append(multistart_r2r4(traj_df_i, duration, starts, cell, timebins, twod=twod))
            print(f'Computed split #{i+1} of {split}.')
elif option == 'eb' :
    out = multiduration_r2r4(traj_df, duration, starts if timebins is None else timebins, cell)
elif option == 'vanhove' :
    out = multistart_r2r4(traj_df, duration, starts, cell, timebins, do_avg=False).reset_index()
elif option == 'dx' :
    out = multistart_r2r4(traj_df, duration, starts, cell, timebins, do_avg=False, col='dx').reset_index()
elif option == 'dy' :
    out = multistart_r2r4(traj_df, duration, starts, cell, timebins, do_avg=False, col='dy').reset_index()

## running an arbitrary custom function is not implemented
# else :
#     func = lambda x: x**2
#     if 'func' in keys: func = eval(options['func'])
#     out = multistart_apply(traj_df, duration, starts, func)
    

## write to a file - unless splitting, in which case write to multiple files
try : out.to_csv(options['file_out'],index=False, float_format='%.7g')
except :
    print('could not save one output file')
    if outs: 
        suffix = f'{duration}ps.csv'
        prefix = options['file_out'].replace(suffix,'')
        for s in range(split) :
            fout = prefix + f'{s+1}-' + suffix
            outs[s].to_csv(fout,index=False, float_format='%.7g')

dtt = (dt.now()-t1).total_seconds()
# print(f'wrote output to {options["file_out"]} at {dt.now().strftime("%y%m%d %H:%M:%S")}')
print(f'file {options["file"]} done, time taken: {dtt:.2f} sec')

try : 
    del traj_df, out
    # print('Deleted trajectory & output from memory.')
except : pass
