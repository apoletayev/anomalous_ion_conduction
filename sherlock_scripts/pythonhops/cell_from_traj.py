#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:46:57 2020

Grabs the cell parameters (orthogonal cells) from a LAMMPS trajectory.

Command-line options required:
    - file_in  : this is the trajectory
    - file_out : this is where the cell gets written

@author: andreypoletaev
"""

from sys import argv
from numpy import array, eye, savetxt

# =============================================================================
# %% read the first few lines of a trajectory and save the cell dimensions
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0], x.split('=')[1]) for x in argv[1:] ])
keys = list(options.keys())

assert 'file_in'  in keys, 'pass file_in=...  [path] as a command-line option'
assert 'file_out' in keys, 'pass file_out=... [path] as a command-line option'

with open(options['file_in'], 'r') as fin:
    
    for i in range(5) : l = fin.readline()
    
    cell = eye(3)
    
    for i in range(3) :
        l = fin.readline()
        d = array(l[:-1].split(' ')).astype(float)
        cell[i,i] = max(d) - min(d)
        
    savetxt(options['file_out'], cell, delimiter=' ', fmt='%.7g')