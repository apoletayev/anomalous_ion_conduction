#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 01:12:18 2020

Testing methods using the newest (2.2.0) version of freud

@author: andreypoletaev
@editor: avrumnoor
"""
# =============================================================================
# %% initial imports
# =============================================================================

import sys, os
if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import pandas as pd
import numpy as np
import crystal_utils as cu
import networkx as nx

import freud, random

from matplotlib import pyplot as plt

# =============================================================================
# %% Write a LMP file for LAMMPS input using a DL_POLY CONFIG as starting point
# =============================================================================
## starting input .CONFIG file with atoms
prod_file = './templates/bdp_template_12101.CONFIG'  ## starting input file

## parameters: metal_input is in the starting file; metal_output gets written.
metal_input  = 'Ag'
metal_output = 'Na'

## defect atom, typically 'Mg'. 'Li' works too.
defect = 'Mg'

## rule for placing defects: 'unsym', 'symm', 'all', 'Al4'
rule = 'unsym'
enforce_d = True if rule == 'symm' else False

## exclude: minimum enforced distance between defects
# excl = 1 if defect == 'Li' else 0
excl = 0

## reset seed once for the whole run. seed=10 was used for "normal" beta-doubleprime
random.seed(10)

## ===== parameters above, loading file below =====

## create a dummy output directory
to_dir = f'./structures/{metal_output} {rule}_{excl} {defect}-doped/'
if not os.path.exists(to_dir): os.makedirs(to_dir)

## read the input .CONFIG file
phase, intro_line, cell, atoms = cu.read_poly(prod_file, fractional=False)

## set a shorthand, this is only for fractional=False just above.
Lx = cell[0, 0]
Ly = cell[1, 1]
Lz = cell[2, 2]

## find the dimensions of the file. 
dimension = [5.630, 9.7514/2, 33.45]
x_dimension = cell[0, 0] / dimension[0]
y_dimension = cell[1, 1] / dimension[1]
z_dimension = cell[2, 2] / dimension[2]

## Find all z-coordinates for conduction planes in the input file: this outputs a list of z-coordinates.
## The z coordinate is now either fractional (from -0.5 to +0.5) or real (angstroms)
planes = cu.get_conduction_planes(atoms, metal_input, inexact=True)

## Find the number of Al atoms that will be randomly substituted by Mg.
## num_sub_al = "Number of (will be) substituted aluminium".
num_defects_to_be = int( cu.find_no_sub_al(atoms, x= 1/3 if defect == 'Li' else 2/3) / len(planes) )

## add symmetry to the atoms, e.g. Al1, Al2, etc.
new_atoms = cu.add_symmetry_bdp(atoms,cell, metal_input, frac=False)
al4_atoms = new_atoms.query('symm == "Al4"')
al2_atoms = new_atoms.query('symm == "Al2"')

## select all possible sites for defect placements, celled defect_sites
if rule == 'all' :
    defect_sites = pd.concat([al4_atoms.copy(deep=True), al2_atoms.copy(deep=True)])
elif rule == 'unsym' : 
    al2a_zs = atoms.query('symm == "Al2"').sort_values(by='z').z.unique()[1::2]
    # print(f'z coordinates of unsymmetric Al(2) sites: {al2a_zs}')
    defect_sites = al2_atoms.query('z in @al2a_zs')
elif rule == 'symm' :
    defect_sites = al2_atoms.copy(deep=True)
elif rule == 'Al4' : 
    defect_sites = al4_atoms.copy(deep=True)
    
## split defect sites into blocks by querying their nearest neighbors 
## then make a graph & separate connected components

## query defect sites for their 3-6 NN's
box = freud.box.Box.from_matrix(cell)
query_args = dict(mode='nearest', num_neighbors=3 if rule == 'symm' else 6, exclude_ii=True)
que = freud.locality.AABBQuery(box, defect_sites[['x','y','z']].values)
result = que.query(defect_sites[['x','y','z']].values, query_args)

## manipulate result to be an edgelist
edges = list()
for r in result :
    if (r[1], r[0]) not in edges: edges.append((r[0], r[1]))

## create graph of possible defect sites, nxg has connected components = blocks
nxg = nx.from_edgelist(edges)
print('#s of possible defect sites per block:', [len(c) for c in nx.connected_components(nxg)])

## separate graph nxg into its blocks using nx.connected_components(nxg)
## and then create defects in each block
for sg in [nxg.subgraph(c).copy() for c in nx.connected_components(nxg)] : 
    
    ## pick nodes of the subgraph that will be substituted with a defect
    picked, _, past = cu.pick_nodes_from_graph(sg.edges(), num_defects_to_be, 
                        exclude=excl, verbose=False, enforce_odd=enforce_d)
    
    ## actually substitute the defect for Al in the picked nodes 
    defect_indices = defect_sites.iloc[list(picked)].index.values
    new_atoms.loc[defect_indices, 'atom'] = defect
    new_atoms.loc[defect_indices, 'symm'] = defect
        
## remove some mobile ions to make stoichiometry work
for z in planes :
    
    mobiles = new_atoms.query(f'atom == @metal_input & {z-2} < z < {z+2}')
    mobile_indices = mobiles.index.values.tolist()
    indices_to_remove = list()
    
    for i in range( int(len(mobiles) / 6) ):
        to_remove = random.choice(mobile_indices)
        mobile_indices.remove(to_remove)
        indices_to_remove.append(to_remove)
    new_atoms.drop(indices_to_remove, inplace=True)
    
## reorder indices s.t. they start at 1 & are consecutive as LAMMPS requires
new_atoms.sort_values(by='idx',axis=0,inplace=True)
new_atoms.reset_index(drop=True, inplace=True)
new_atoms.index += 1

## change the name of the mobile ion
metal_indices = new_atoms.query('atom == @metal_input').index.values.tolist()
new_atoms.loc[metal_indices,'atom'] = metal_output

## write files: CONFIG first, then LMP

## write the DL_POLY .CONFIG file - for visualization
cu.write_poly(to_dir+f'{metal_output}_bdp_{rule}_{excl}.CONFIG', phase, intro_line, cell, new_atoms, fractional=False)

## write the LAMMPS file
cu.write_lmp(to_dir+f'{metal_output}_bdp_{rule}_{excl}.lmp', cell, new_atoms, 
             fractional=False, defects=None)

# =============================================================================
# %% DEBUG/QC: extract and re-create grids from written LAMMPS files
# ## the grid files that my 2019 code uses are vertices of the polygons,
# ## of which the centers are sites of the mobile ions (BR and aBR)
# =============================================================================
metal = 'Na'
defect = 'Mg'

## for quicker testing
rule = 'unsym'
# excl = 0 if (rule == 'c' or rule == 'b') else 1
excl = 0

## flag to load the atoms with fractional coordinates (if True)
frac = False

## whether to do plots
viz=True

## this is where the grids will be written
to_dir = f'./structures/{metal} {rule}_{excl} {defect}-doped/'

## try importing a python-generated .lmp file
fn = to_dir + f'{metal}_bdp_{rule}_{excl}.lmp'  ## this should be carried over from above
_, _, cell, atoms = cu.read_lmp(fn, fractional=frac)  ## import everything that is not interstitial oxygens
Lx, Ly, Lz = np.diag(cell) if not frac else np.ones(3)

## shorthand
mgs = atoms.query('atom == @defect')

## Find all z-coordinates for conduction planes in the input file: this outputs a list of z-coordinates.
## The z coordinate is now either fractional (from -0.5 to +0.5) or real (angstroms)
planes = cu.get_conduction_planes(atoms, metal, inexact=True)

## keep track of how far apart planes are
delta_z = np.mean(np.diff(planes)) 

## count all the #'s of sites
max_dist = 8
all_counts_0 = np.zeros(max_dist)
all_counts_1 = np.zeros(max_dist)
all_counts = np.zeros(max_dist)

## Loop through the conduction planes
for i, plane in enumerate(planes):
    
    ## split conduction plane into mobile-ion sites just-above and just-below    
    mobile_ion_sites, sites_below, sites_above, path_lengths \
        = cu.get_sites_above_below(plane, atoms, cell if not frac else np.eye(3), 
                                   metal=metal, frac=frac, viz=viz)
    
    num_sites = len(mobile_ion_sites)
            
    ## select Mg sites next to this plane. Both sides are needed. 
    ## Furthermore, it is necessary to wrap around the boundary of the cell
    nearby_mg = mgs.query(f'{plane - delta_z} < z < {plane + delta_z}')
    # print(f'found nearby Mg atoms: {len(nearby_mg)}')
    
    ## accout for wrapping around the cell boundary
    if Lz*0.5 - abs(plane) < delta_z:
        nearby_mg2 = mgs.query(f'z < {-Lz + abs(plane) + delta_z} or z > {Lz - abs(plane) - delta_z}')
        # print(f'found wrapped Mg atoms: {len(nearby_mg2)}')
        nearby_mg = pd.concat([nearby_mg, nearby_mg2]).drop_duplicates()
    
    print(f'found total Mg atoms: {len(nearby_mg)}')
    mg_pts = nearby_mg[['x', 'y', 'z']].values
    
    if viz:
        # plt.gca().scatter(mg_sites.x, mg_sites.y, c='k', s=75)
        plt.gca().set(title=f'z={plane:.3f}', aspect=1)
        plt.gcf().tight_layout()
    
    ## test whether the graph distances from Mg sites re-compute as 0's and 1's
    ## when loaded from a file 
    ## get sites closest to Mg defects. list(set()) suppresses duplicates
    edges_0, edges_1, d0, d1 = cu.get_nearest_points(mobile_ion_sites, mg_pts, cell if not frac else np.eye(3))
    sites_0 = [x[1] for x in edges_0]; mgs_0 = [x[0] for x in edges_0]; counts_0 = list()
    sites_1 = [x[1] for x in edges_1]; mgs_1 = [x[0] for x in edges_1]; counts_1 = list()
    
    ## calculate paths to Mg defects for all sites in this plane
    if len(sites_0) > 0:
        counts_0 = [[sites_0.count(x) for x in range(num_sites)].count(i) for i in range(max_dist)]
        paths_to_mg0 = [min(path_lengths[sites_0, x]) for x in range(num_sites)]
        all_counts_0 += np.array(counts_0)
    else: paths_to_mg0 = np.ones(num_sites)*num_sites
    if len(sites_1) > 0:
        counts_1 = [[sites_1.count(x) for x in range(num_sites)].count(i) for i in range(max_dist)]
        paths_to_mg1 = [min(path_lengths[sites_1, x])+1 for x in range(num_sites)]
        all_counts_1 += np.array(counts_1)
    else: paths_to_mg1 = np.ones(num_sites)*num_sites
        
    # combine path lengths to distance==1 and distance==0 sites taking min()
    paths_to_mg = [min(paths_to_mg0[i], paths_to_mg1[i]) for i in range(num_sites)]
    counts = [[(sites_1+sites_0).count(x) for x in range(num_sites)].count(i) for i in range(max_dist)]

    ## print some distances and counts of defects next to sites
    print(f'Numbers of closest defects at 0: {counts_0}')
    print(f'Numbers of closest defects at 1: {counts_1}')
    print(f'Combined #s of closest defects : {counts}')
    print(f'distances at 0: {np.unique(np.round(d0,4))}')
    print(f'distances at 1: {np.unique(np.round(d1,4))}')
    all_counts += np.array(counts)
    
    ## visualize Mg's at specific distances : 
    ## both in real space & graph separation
    mgs_0_long = [ mgs_0[i] for i in np.where(6.2 < np.array(d0) )[0] ]
    mgs_1_long = [ mgs_1[i] for i in np.where(6.2 < np.array(d1) )[0] ]
    mgs_0_short = [ mgs_0[i] for i in np.where( np.array(d0) < 6.2)[0] ]
    mgs_1_short = [ mgs_1[i] for i in np.where( np.array(d1) < 6.2)[0] ]
    
    ## plot sites above and below
    if viz:
        plt.gca().scatter(sites_below[:,0], sites_below[:,1], c='r', s=100, alpha=0.5, label='lower site')
        plt.gca().scatter(sites_above[:,0], sites_above[:,1], c='r', s=200, alpha=0.5, label='upper site')
        
        ## plot where the Mg are by how they are classified
        plt.gca().scatter(mg_pts[mgs_0_long,0], mg_pts[mgs_0_long,1], c='b', s=60, label='0NN Mg, far')
        plt.gca().scatter(mg_pts[mgs_0_short,0], mg_pts[mgs_0_short,1], c='g', s=40, label='0NN Mg, close')
        plt.gca().scatter(mg_pts[mgs_1_long,0], mg_pts[mgs_1_long,1], c='tab:orange', s=75, label='1NN Mg, far')
        plt.gca().scatter(mg_pts[mgs_1_short,0], mg_pts[mgs_1_short,1], c='k', s=30, label='1NN Mg, close')
        
        plt.gca().set(xlim=[-0.55*Lx, 0.55*Lx], ylim=[-0.55*Ly, 0.55*Ly])
    
        plt.gca().legend(ncol=3)

print(f'===== Totals for placement={rule} exclude={excl} =====')
print(f'Numbers of closest defects at 0: {all_counts_0}')
print(f'Numbers of closest defects at 1: {all_counts_1}')
print(f'Combined #s of closest defects : {all_counts}') 
    


    
    
