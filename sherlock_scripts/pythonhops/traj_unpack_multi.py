#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 2 12:28:25 2019

USAGE  : python3 traj_unpack_multi.py trajectory_path=... folder_out=... first_atom=1 last_atom=500 metal=Na avg=False type=1
EXAMPLE: python3 traj_unpack_multi.py filepath=./trajectories/Na_120_1000Ktraj2.dat folder_out=./atoms/Na120_1000K/ first_atom=1 last_atom=12000 atom=Na

Required Parameters:
filepath        : path(s) to the lammps output, no lists taken, only * as wildcard
folder_out      : folder where the atoms will go (it must exist)
atom            : chemical name of the ion getting parsed, e.g. Na

Optional Parameters:
first_atom      : index of the first atom, default is 1
last_atom       : index of the last atom (OK if too large - only used as cutoff), default is 12000
avg (averaging) : whether to output an average (yes, no, only, True, False). Default is False.
max_time        : [picoseconds] if provided, stop at this value of time. 
timestep        : [picoseconds] the timestep of the simulation in case it is not 1 fs.
echo            : [picoseconds] output status updates every this many psec.

The atoms that will be parsed are a subset of all atoms: 
    - LAMMPS type given by the last input parameter. Default is 1.
    - indices between first atom & last atom from the input. Default is 1-12000.

## Edits 2020/06/13 : moved the %-doneness output to timestep mode

## Edits 2020/06/13 : automated which files are opened rather than bulk open-all
                      in the beginning for non-consecutively numbered atoms
                      
## Edits 2020/07/19 : removed all explicit counting, added time-doneness
                      added filter by type of atom
                      
## Edits 2020/10/26 : added support for multiple files with wildcard * in path
                      removed support for multiple types of atoms

@author: andreypoletaev
"""

import numpy as np
#import pandas as pd
from datetime import datetime as dt
#import hop_utils as hu
import sys

import glob

DAT = '.dat'


t_step = 0.001

################################################################################
#%% convert string to int or float
### function defined here such as to avoid import commands
################################################################################

def s2n(s):
    '''
    check if string is a number and convert it to a number if so
    '''
    try :
        return int(s)
    except ValueError:
        try :
            return float(s)
        except ValueError:
            return False

##############################################################################
#%% trajectory import line by line because file too long (typically >3GB)
##############################################################################

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0],x.split('=')[1]) for x in sys.argv[1:] ])

keys = list(options.keys())

assert 'filepath' in keys and 'folder_out' in keys and 'atom' in keys, \
    'pass filepath=... [path] , folder_out=... [folder] , atom=... [str] as command-line options'

filepath = options['filepath']
folder_out = options['folder_out']
metal = options['atom']

# ===== read the optional parameters =====
try: first_atom = eval(options['first_atom'])
except: first_atom = 1

try: last_atom = eval(options['last_atom'])
except: last_atom = 12000

try: avg = eval(options['avg'])
except: avg = False

try: echo = eval(options['echo'])
except: echo = 10

## default timestep is 1 fs, this is [picoseconds]
try: 
    ## round to 3 places to avoid floating-point errors
    t_step = np.round(eval(options['timestep']), 3)
    print(f'WARNING: Using a non-default timestep: {t_step} ps')
except : t_step = 0.001

## Set which type of atom is getting parsed
## This is hard-coded and will need to get modified if another material is used.
## This will also give wrong results if e.g. Li is used as a defect in beta".
if metal in ['Na', 'K', 'Ag', 'Li'] : atom_type=1
elif metal in ['Al'] : atom_type = 2
elif metal in ['O','Ob','Op'] : atom_type = 3
elif metal in ['Oi','Mg'] : atom_type = 4
else :
    metal = 'Na' ## default
    atom_type = 1 ## default
    
fo_dict = dict()
num_atoms = 0

if (avg == 'yes') or (avg == 'only') or (avg == True) or (avg == 1) : 
    fo_avg = open(folder_out + metal + 'avg.dat','a+')
    
## long-time cutoff : 
try : 
    max_time = eval(options['max_time'])
    
except : max_time = False
    
# =============================================================================
# %% check for multiple trajectory files and their first time steps
# =============================================================================

if '*' in filepath:
    paths = sorted(glob.glob(filepath))
    
    if not paths : raise IOError(f'Filepath yields no files: {filepath}')
    if len(paths) == 1 : 
        paths = [filepath.replace('*','')]
        first_timesteps = [0]
    else:
        first_timesteps = list()
        for path in paths :
            with open(path, 'r') as fp:
                mode = None
                line = fp.readline()
                while line :
                    arr = line[:-1].split(' ') ## take out the \n
                    first = arr[0]
            
                    if 'ITEM' in first: mode = arr[1]
                    elif mode == 'TIMESTEP' : 
                        first_timesteps.append(s2n(first))
                        break
                    line = fp.readline()
                    
            print(f'first timestep {first_timesteps[-1]} in file {path}', flush=True)
    
else :
    paths = [filepath]
    first_timesteps = [0]

################################################################################
# %% BIG LOOP over all lines in trajectory file
################################################################################

path_number = 0
fp = open(paths[path_number], 'r') 
print(f'\nstarting to parse file {paths[path_number]}.', flush=True)
    
## initial conditions and counters
ts = 0 ## this is timestamp in [step] as recorded by LAMMPS, usually [fs]
mode = None
line = fp.readline()
dt0 = dt.now()

while(line) :
    arr = line[:-1].split(' ') ## this takes out the newline character
    first = arr[0]
    
    if 'ITEM' in first: ## line has words, not numbers, so reset mode
        mode = arr[1]
        ## write stuff to file(s) if formatting
        ## consider writing headers if this is 'ITEM: ATOMS'
    elif mode == 'TIMESTEP' :
        ts = s2n(first)
        
        if (ts * t_step).is_integer() and (ts * t_step) % echo == 0 :
            print(f'{ts / 1000.} ps done, {(dt.now() - dt0).total_seconds():.2f} sec', flush=True)
            
        ## if averaging, then divide by the count of atoms and write
        if (ts > 0.1) and (avg in ['only', 'yes', True, 1]) and (num_atoms > 0 or len(fo_dict) > 0) :
            num_atoms = max(num_atoms,len(fo_dict))
            fo_avg.write(str(ts * t_step)); fo_avg.write(' ')
            fo_avg.write(' '.join([f'{x/num_atoms:.6f}' for x in this_timestep_avg]) + '\n')
            
        ## switch to the next input file if needed and if possible
        if (ts in first_timesteps) and first_timesteps.index(ts) > path_number:
            path_number = first_timesteps.index(ts)
            fp.close()
            fp = open(paths[path_number], 'r')
            print(f'\n at time step {ts}, starting to parse file {paths[path_number]}.', flush=True)
            fp.readline()
            fp.readline() ## there could be a test loop here to check the timestep
        
        ## stop if max time is defined and reached
        if max_time and ts * t_step > max_time:
            print(f'Reached the provided time cutoff: {max_time} ps. Stopping here.', flush=True)
            break
            
        ## write stuff to file(s) if formatting
    elif mode == 'NUMBER' :
        ## number of atoms
        ## write stuff to file(s) if formatting
        pass
    elif mode == 'BOX' :
        # print([float(a) for a in arr])
        # print(float(arr[1]) - float(arr[0]))
        pass
        ## write stuff to file(s) if formatting for not just python
    elif mode == 'ATOMS' :
        # check that the atom is a needed one
        if (first_atom <= s2n(first) <= last_atom) and (int(arr[1]) == atom_type) :  ## this is a needed atom
            
            ## replace the type of atom with time
            arr[1] = str(ts * t_step)
            
            ## open file if one does not exist; also needed for counting.
            ## this starts a new file as it would be a pain to check the time
            ## stamps on ALL files from a trajectory - which would likely not 
            ## all be the same. 
            if s2n(first) not in list(fo_dict.keys()) and avg != 'only' :
                fo_dict[s2n(first)] = open(folder_out + metal + first + DAT,'w+')
                print(f'started new file for atom {s2n(first)}, {len(fo_dict.keys())} total')
            
            ## count atoms once
            elif avg == 'only' and ts == 0 : 
                num_atoms += 1 
            
            ## write the line for the single atom
            if avg != 'only' : 
                fo_dict[s2n(first)].write(' '.join(arr) + '\n')
            
            ## averaging: add up positions or stuff in "this timestep's average"
            if avg in ['only', 'yes', True, 1] :
                if ts == 0 :
                    if num_atoms == 1 : ## this is the very first atom
                        first_atom = s2n(first)
                    elif len(fo_dict) == 1:
                        first_atom = s2n(first)
                if s2n(first) == first_atom : ## start a new array
                    this_timestep_avg = np.array([s2n(x) for x in arr[2:]])
                else :
                    this_timestep_avg += np.array([s2n(x) for x in arr[2:]])
        else: 
            ## not one of the needed atoms
            pass
    else :
        print('unclear mode:', mode)
    
    line = fp.readline()

## close output files
if avg != 'only' : 
    for fo in list(fo_dict.values()) : 
        fo.close()
if avg in ['only', 'yes'] : 
    fo_avg.close()
