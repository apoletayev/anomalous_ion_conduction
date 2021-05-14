#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 01:41:52 2020

Combines LAMMPS output files coming from a series of restarts with a * wildcard.
This works on expanded (mode scalar) fixes from LAMMPS where each line is a time.

The overlapping values of times due to restarts are averaged, but they should be identical.

Required command-line args : filenames= ,
Optional command-line args : file_out= ,

@author: andreypoletaev
"""

# =============================================================================
# %% Imports and constants
# =============================================================================

import pandas as pd

import sys

from glob import glob

# =============================================================================
# %% parse input and combine
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0],x.split('=')[1]) for x in sys.argv[1:] ])

keys = list(options.keys())

# print(options)

assert 'filenames' in keys, 'please pass filenames=... [path] as command-line option'

# template = f'/*_vacf_{int(options["duration"])}ps.csv' if 'template' not in keys else options['template']

file_out = options['filenames'].replace('*','') if 'file_out' not in keys else options['file_out']

print('looking for files that look like this: '+options['filenames'], flush=True)

output = pd.DataFrame()
counter = 0

files_to_combine = sorted(glob(options['filenames']))

assert len(files_to_combine) > 1, 'Only one file fits the bill, skipping combining.'
print(files_to_combine, flush=True)

for fin in files_to_combine:
        
    try: 
        ## read the header for column names
        fp = open(fin, 'r')
        line1 = fp.readline()
        line2 = fp.readline()
        fp.close()
        
        colnames = line2[:-1].split(' ')[1:]
        
        ## read the actual numbers
        df = pd.read_csv(fin, skiprows=1, sep=' ')
        # colnames = df.iloc[0,1:-1].tolist()
        df = df.iloc[:, :-1]
        df.columns = colnames
        
        df = df.apply(pd.to_numeric)
        
        # print(df.columns)
        # print(df.head(5))
        # print(df.dtypes)
        
        # print(df.head())
        if len(df) > 0:
            output = output.append(df, ignore_index=True)
            counter += 1
            print(f'appended data from file #{counter} : {fin}', flush=True)
    except: print(f'could not load / add {fin}', flush=True)
    
## ensemble-average in all cases - but not always the first thing
output = output.groupby('TimeStep').agg('mean').reset_index().rename(columns={'TimeStep':line1[:-1]+'\n# '+'TimeStep'})
# output.TimeStep = output.TimeStep.astype(int)
    
## write file normally
output.to_csv(file_out, index=False, float_format='%.6g', sep=' ')