#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:55:08 2020

@author: andreypoletaev

Assumptions made: 
    time is in picoseconds, timestep is 1 fs
    
"""

# =============================================================================
# %% Imports & constants
# =============================================================================

import sys

from hop_utils import autocorrelation

import pandas as pd

## column names for the cases that the file is a CoM file or a single-atom velocity file
com_col_names = ['timestep', 'x', 'y', 'z', 'vx', 'vy', 'vz']
vel_col_names = ['atom_id', 'time', 'vx', 'vy', 'vz']

# =============================================================================
# %% Parse inputs 
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0],x.split('=')[1]) for x in sys.argv[1:] ])

# print(options)

assert 'file' in list(options.keys()) and 'duration' in list(options.keys()), \
    'please pass file=... [path] and duration=... [psec] as command-line options'
    
col_names = vel_col_names
header = 0

if ('com' not in list(options.keys())) or (eval(options['com']) == True) :
    col_names = com_col_names
    header = 2
    
fin = pd.read_csv(options['file'], sep=' ', skiprows=header, names=col_names, index_col=False)

# print(fin.head(5))

## convert time from [steps] to [ps] if the input file has the former
try : fin['time'] = fin.timestep / 1000. ## hard-coded conversion from steps to picoseconds
except : pass

fin.set_index('time', inplace=True)

# folder = '/'.join(options['file'].split('/')[:-1])
# fn = options['file'].split('/')[-1]
dur = int(options['duration'])
fout = options['file_out']

## do the actual computation of the autocorrelation function
print(f'computing {options["file"]}')
jacf = autocorrelation(fin, dur, ['x','y','z'], verbose=True, to_file=fout).reset_index().rename(columns={'index':'time'})
# jacf.to_csv(folder+'/'+fn[3:-4]+f'_{dur}ps.csv', index=False)
print(f'computed and saved {options["file"]}')