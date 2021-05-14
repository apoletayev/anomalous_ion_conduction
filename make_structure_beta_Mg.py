#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 01:12:18 2020

Testing methdos using the newest (2.2.0) version of freud

@author: andreypoletaev
"""

# =============================================================================
# %% Imports
# =============================================================================

import sys, os, freud, random
if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import pandas as pd
import numpy  as np
import crystal_utils as cu
import networkx as nx

from matplotlib import pyplot as plt

# =============================================================================
# %% Generate a structure for Mg-doped beta-alumina
# ## random seed for 120_M4 : 748
# ## random seed for 120_M1 : 2
# =============================================================================
## starting input .CONFIG file with atoms
## this format has coordinates from -L/2 to +L/2, not 0 to L
prod_file = './templates/stNabeta_ortho10102.CONFIG'

## parameters and constants
metal = 'Na'

random.seed(748)  ## reset the random seed.
exclude = 4  ## minimum network distance between Mg to enforce when picking 

## Example 1.20 stoichiometry : this is the # of interstitials in every plane
num_mgs_per_plane = 20
packed = False

## subselect for an un-symmetric distribution of Al(2)'s if True, (Davies 1986)
subselect = False

# ===== automatic things below this line =====

## flag for importing atoms with fractional (True) or real-space (False) coordinates
frac = False

## calculate stoichiometry of the resulting phase for naming the output files below.
stoich = num_mgs_per_plane + 100

to_dir = f'./structures/{metal} {stoich}_{exclude} Mg-doped/' ## directory to which grids, files & distances are written
if not os.path.exists(to_dir): os.makedirs(to_dir)

## read the input .CONFIG file
phase, intro_line, cell, atoms = cu.read_poly(prod_file, fractional=frac)

## find all conduction planes in the input file
planes = cu.get_conduction_planes(atoms, metal)
dz = np.mean(np.diff(planes))

## initialize data structure with symmetry elements
new_atoms = cu.add_symmetry_beta(atoms, cell, mobile='Na', frac=frac)

## create freud query with Al2 points
al2s = new_atoms.query('symm == "Al2"')

## subselect Al(2)'s for unsymmetric placement: 
if subselect:
    al2zs = np.sort(al2s.z.unique())
    top_bottom = cell[2,2]+al2zs[0]-al2zs[-1]
    
    zdiffs = np.append(np.diff(np.sort(al2s.z.unique())), top_bottom)
    
    if top_bottom < np.mean(zdiffs) :
        al2s = al2s.query(f'z in {list(al2zs[1::2])}')
        print("top & bottom Al(2)'s are one block")
    else :
        al2s = al2s.query(f'z in {list(al2zs[::2])}')
        print("top & bottom Al(2)'s are different blocks")

print(f'{al2s.atom.count()} total Al(2) atoms substitutable')
    
## query Al2 sites for their 3 NN's
box = freud.box.Box.from_matrix(cell)
query_args = dict(mode='nearest', num_neighbors=6 if subselect else 3, exclude_ii=True)
que = freud.locality.AABBQuery(box, al2s[['x','y','z']].values)
result = que.query(al2s[['x','y','z']].values, query_args)

## manipulate result to be an edgelist
edges = list()
for r in result :
    if (r[1], r[0]) not in edges: edges.append((r[0], r[1]))

## create graph of Al2 sites
## nxg should have one connected component per block
nxg = nx.from_edgelist(edges)
# print([len(c) for c in nx.connected_components(nxg)])

## separate graph nxg into its blocks
for sg in [nxg.subgraph(c).copy() for c in nx.connected_components(nxg)] : 
    # print(len(sg), len(sg.edges()))
    
    ## pick nodes that will be substituted with Mg
    picked, _, past = cu.pick_nodes_from_graph(sg.edges(), num_mgs_per_plane, 
                        exclude=exclude, verbose=False, enforce_odd=False)
    
    ## substitute Mg for Al in the picked nodes 
    defect_indices = al2s.iloc[list(picked)].id
    new_atoms.loc[defect_indices, 'atom'] = 'Mg'
    new_atoms.loc[defect_indices, 'symm'] = 'Mg'
    
## create a placeholder for new mobile ions to be added
newatom_list = list()

## for every plane, place extra mobile ions.
for i, pl in enumerate(planes):
    
    ## find the empty sites
    empties, _ = cu.find_empty_brabr(atoms, pl, cell)
    empties = pd.DataFrame({'x': empties[:, 0], 'y': empties[:, 1], 'z': empties[:, 2]})
    max_idx = new_atoms.id.max()
    
    ## place extra mobile ions on random empty sites 
    for j,s in zip(range(i*num_mgs_per_plane +1,(i+1)*num_mgs_per_plane+1), \
                   random.sample(range(len(empties)), num_mgs_per_plane)) :
        s = empties.iloc[s]
        newatom_list.append({'idx': max_idx + j, 'atom': metal, 'x': s.x, 
                             'y': s.y, 'z': s.z, 'symm':'NA', 'id':max_idx+j})
        
## actually add all the new mobile ions at the same time
new_mobiles = pd.DataFrame(newatom_list).set_index('idx').sort_index()
print(f'adding {len(new_mobiles)} atoms with indices {new_mobiles.index.min()}-{new_mobiles.index.max()}')
new_atoms = new_atoms.append(new_mobiles)
    
## write CONFIG file
cu.write_poly(to_dir+f'{metal}beta{stoich}_M{exclude}.CONFIG',phase, intro_line, cell, new_atoms, fractional=frac)

## write LAMMPS file
cu.write_lmp(to_dir+f'{metal}beta{stoich}_M{exclude}.lmp', cell, new_atoms, defects=None, fractional=frac)

# =============================================================================
# %% DEBUG/QC: extract and re-create grids from written LAMMPS files 
# ## the grid files that my 2019 code uses are vertices of the polygons,
# ## of which the centers are sites of the mobile ions (BR and aBR)
# =============================================================================

stoich = 120  ## from above
exclude = 4
metal = 'Na'  ## from above / default. All code uses Na as switching is easiest in the .lmp file 
frac = False  ## flag to load the atoms with fractional coordinates (if True)

## this is where the grids will be written
to_dir = f'./structures/{metal} {stoich}_{exclude} Mg-doped/'

## try importing a python-generated .lmp file
fn = to_dir + f'{metal}beta{stoich}_M{exclude}.lmp'  ## this should be carried over from above
_, _, cell, atoms = cu.read_lmp(fn, ignore=[4], fractional=frac)  ## import everything that is not interstitial oxygens
_, _, _, inters = cu.read_lmp(fn, ignore=[1, 2, 3], fractional=frac)  ## import interstitial oxygens separately
all_ois = inters.copy(deep=True)

Lx, Ly, Lz = np.diag(cell) if not frac else np.ones(3)  ## box size for freud.

## find the conduction planes
planes = cu.get_conduction_planes(atoms, metal)

## hard-coded names for planes. Will be different for beta-doubleprime
plane_names = ['{:03d}'.format(x) for x in ((planes/Lz+0.5) * 100).astype(int)]

## iterate through planes
for pl, pname in zip(planes, plane_names):
    
    ## get and plot the mobile-ion sites for this plane
    mobile_sites = cu.get_mobile_ion_sites(atoms, pl, cell)
    _, edges, _ = cu.get_mid_oxygen_sites_freud(mobile_sites, cell, viz=False)
    
    ## try auto-getting BR sites
    site_types = cu.auto_get_BR_sites(atoms, cell, mobile_sites,atoms_are_frac=frac)
    BR_sites = [i for i in range(len(site_types)) if site_types[i] == 'BR']
    
    fig, ax = plt.subplots()
    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D = True)
    site_vor = freud.locality.Voronoi(box)
    mobile_sites[:,-1] = 0
    site_vor.compute((box, mobile_sites))
    
    cu.draw_voronoi(box, mobile_sites, [site_vor.polytopes[i] for i in BR_sites],cell_numbers=BR_sites)
    plt.gca().set(aspect = 1 if not frac else cell[1,1]/cell[0,0], title=f'z={pname}')
    plt.gcf().tight_layout()
    
    ## plot interstitial oxygens: this better match what vesta shows
    if stoich > 100:
        ois = all_ois.query('@pl == z')  ## if loading directly from .lmp
        plt.gca().scatter(ois.x, ois.y, c='red', s=60)
    
    

























