#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:01:08 2021

Command-line options required:
    - atoms  : crystal structure
    - template : prefix to the filenames of the by-plane parsed hops (inc. folder)

@author: andreypoletaev
"""

from sys import argv
# import numpy as np

from glob import glob

from crystal_utils import read_lmp, get_conduction_planes, get_mobile_ion_sites

from hop_utils import which_one_in, combine_planes3

# from pandas import read_csv 

mobile_ions = ['Na', 'Ag', 'K']
host_ions = ['Al', 'O', 'Mg']
phases = ['beta', 'bdp']

## flag, True if correcting velocities
vel = False

## placeholder for the name of the output file
fout_name = None

# =============================================================================
# %% read inputs, find planes
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0], x.split('=')[1]) for x in argv[1:] ])
keys = list(options.keys())

assert 'atoms'  in keys, 'pass atoms=...  [path] as a command-line option'
assert 'template' in keys, 'pass template=... [path] as a command-line option'

prefix = options['template']

print('Combining planes...')
_, _, cell, atoms = read_lmp(options['atoms'])
if len( atoms.query('atom == "O"').type.unique() ) > 1:
    type_ointer = atoms.query('atom == "O"').type.max()
    atoms = atoms.query('type != @type_ointer')
            
mm = which_one_in(mobile_ions, atoms.atom.unique())
ph = which_one_in(phases, options['atoms'])
mobile = atoms.query('atom == @mm')
print(f'Metal: {mm}, phase: {ph}')

assert ph is not None, 'unknown phase, cannot combine planes'

planes = get_conduction_planes(atoms,mm, inexact = (ph == 'bdp'))
site_pts = get_mobile_ion_sites(atoms, planes[0], viz=True)
polys = len(site_pts)

plane_hops = sorted(glob(f'{prefix}*.csv'))
zs_list = [ x.replace('plane.csv','').replace(prefix,'').replace('_','') for x in plane_hops]

print(f'planes: {planes}')
print(f'indices: {zs_list}')
print(f'{polys} sites per plane, {len(mobile)/len(zs_list)} mobile ions per plane')

if len(plane_hops) == len(planes):
    ## looks like hops in all planes are available. combine planes 
    # print(f"Combining planes for {this_file} at T1 = {TK:4d}K.")
    # planes_list = T1_planes.hop_path.values.tolist()
    # planes_folder = '/'.join(planes_list[0].split('/')[:-1])
    # zs_list = T1_planes.z.values.tolist()
    combined = combine_planes3(plane_hops, zs_list, numpolys=polys, verbose=True)
    combined_path = prefix + '_z_allplane.csv'
    combined.to_csv(combined_path, index=False)
else :
    print(f'{len(plane_hops)} files with hops, {len(planes)} planes in .lmp file?!')















