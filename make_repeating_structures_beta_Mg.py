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
from datetime import datetime as dt

plt.rc('legend', fontsize=12)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# =============================================================================
# %% Generate a bunch of random structures and count their sites
# =============================================================================
## starting input .CONFIG file with atoms
prod_file = './templates/stNabeta_ortho10102.CONFIG'

## parameters and constants
metal = 'Na'

## The count of extra mobile ions per plane; e.g.: 20 for 1.20 stoichiometry
defects_per_plane = 20

## subselect for an un-symmetric distribution of Al(2)'s if True, (Davies 1986)
subselect = False

## read the input .CONFIG file & add symmetry elements
phase, intro_line, cell, atoms = cu.read_poly(prod_file, fractional=False)
atoms = cu.add_symmetry_beta(atoms, cell, mobile=metal, frac=False)

## create a separate dataframe with Al2 atoms
al2s = atoms.query('symm == "Al2"')

## subselect Al(2)'s for unsymmetric placement: 
## (1) get smallest difference btwn sorted unique z's, pair them
## (2) pair top and bottom ones if sum of their distances to min & max is comparable to (1)
## subselect :
## if top & bottom are a pair: by every other unique value of z, start w/ 2nd
## else: by every other unique value of z, start w/ 2nd

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

print(f'{al2s.atom.count()} total Al(2)')

## query Al2 sites for their 3 NN's (if all Al2's were picked; no subselection)
box = freud.box.Box.from_matrix(cell)
query_args = dict(mode='nearest', num_neighbors=6 if subselect else 3, exclude_ii=True)
que = freud.locality.AABBQuery(box, al2s[['x','y','z']].values)
result = que.query(al2s[['x','y','z']].values, query_args)

## manipulate result to be an edgelist
edges = list()
for r in result :
    if (r[1], r[0]) not in edges: edges.append((r[0], r[1]))

## create graph of Al2 sites; it should have 1 connected component per block
nxg = nx.from_edgelist(edges)

## find all conduction planes in the input file, and the spacing between them
planes = cu.get_conduction_planes(atoms, metal)
dz = np.mean(np.diff(planes))

dict_of_excludes = dict()
start = dt.now()

## 
for exclude in [1,4]:

    ## dictionary of histograms of paths to defect-closest sites
    hists_dict = dict()
    
    ## iterate over random seeds
    for seed in range(1000 if exclude == 4 else 100) :
        
        ## reset seed
        random.seed(seed)
        
        ## start timer for sanity
        seed_start = dt.now()
    
        ## initialize data structures
        new_atoms = atoms.copy(deep=True)
        broken=False
        
        ## paths to defects: 0-away & 1-away
        all_paths_to_defect_0 = list()  
        all_paths_to_defect_1 = list()
        
        ## paths to defects in all planes
        all_paths_to_def = list() 
        
        ## for each block, pick spots & place Mg
        for sg in [nxg.subgraph(c).copy() for c in nx.connected_components(nxg)] : 
            
            ## if a previous block did not work, short-circuit
            if broken : break
            
            ## pick nodes that will be substituted with Mg at random
            picked, _, past = cu.pick_nodes_from_graph(sg.edges(), defects_per_plane, 
                                exclude=exclude, verbose=False, enforce_odd=False)
            
            if len(picked) < defects_per_plane : broken=True; break
            
            ## substitute Mg for Al in the picked nodes 
            defect_indices = al2s.iloc[list(picked)].id
            new_atoms.loc[defect_indices, 'atom'] = 'Mg'
            new_atoms.loc[defect_indices, 'symm'] = 'Mg'
            
        ## create a placeholder for the coordinates of the created defects
        defect_pts = new_atoms.query('atom == "Mg"')[['x', 'y', 'z']].values
        
        ## for every plane, find mid-oxygen sites
        for pl in planes:
            
            ## if something did not work upstream, then short-circuit
            if broken: break
        
            ## get the mobile-ion sites for this plane
            site_pts = cu.get_mobile_ion_sites(atoms, pl, cell)
            
            ## find the graph distances to Mg defects above & below the plane
            
            ## get all the mid-oxygen sites in this plane
            mid_oxs, edges, midpts = cu.get_mid_oxygen_sites_freud(site_pts, cell, viz=False)
            
            ## create a proper networkx graph from site-edge list
            site_graph = nx.from_edgelist(edges)
            path_lengths = cu.path_lengths(site_graph)
            
            ## find the Mg closest to each mobile-ion site in this plane
            e0, e1, d0, d1 = cu.get_nearest_points(site_pts, defect_pts, cell, num_nn=6)
            e0 = np.array(e0)[np.array(d0)<dz]
            e1 = np.array(e1)[np.array(d1)<dz]
            
            ## indices of mobile-ion sites; can be combined with checking max distance
            s0 = [x[1] for x in e0]
            s1 = [x[1] for x in e1]
            
            # it will be more tidy for combining distances later to keep placeholder arrays
            if len(s0) > 0:
                all_paths_to_defect_0.append([min(path_lengths[s0, x]) for x in range(len(site_pts))])
            else:
                all_paths_to_defect_0.append(np.ones(len(site_pts))*len(site_pts))
            if len(s1) > 0:
                all_paths_to_defect_1.append([min(path_lengths[s1, x])+1 for x in range(len(site_pts))])
            else:
                all_paths_to_defect_1.append(np.ones(len(site_pts))*len(site_pts))
                
            # combine path lengths to distance==1 and distance==0 sites taking min()
            this_plane_paths = [min(all_paths_to_defect_0[-1][i], all_paths_to_defect_1[-1][i]) for i in range(len(site_pts))]
            
            all_paths_to_def.append(this_plane_paths)        
            
        if not broken: 
            ## remove nesting in the path lengths to defect sites
            all_paths_to_def = np.array([x for x in cu.flatten(all_paths_to_def)])
            
            ## save the histogram
            hists_dict[seed] = np.histogram(all_paths_to_def, np.arange(-0.5,0.5+max(map(max,path_lengths))))[0]
            
            ## show a sign of life
            print(f'({exclude:2d},{seed:2d}) took {(dt.now()-seed_start).total_seconds():4.1f} sec, \
                  total {(dt.now()-start).total_seconds():4.1f} sec')
        else:
            print(f'({exclude:2d},{seed:2d}) took {(dt.now()-seed_start).total_seconds():4.1f} sec, was broken')
        
    ## dataframe with all distances by seed
    dict_of_excludes[exclude] = pd.DataFrame(data = hists_dict)
    
# =============================================================================
# %% compute z-scores to find the most-average structure
# ## most-average for 120:  ex=M1 seed=2   (max seed=100, no packing, no asymmetry)
# ##                        ex=M4 seed=748 (max seed=1000, no packing, no asymmetry, ~20/1000 work) << this was used
# =============================================================================

min_dist=0
max_dist=10

for exclude, distances in dict_of_excludes.items():
    
    if distances.empty: continue
    
    distances = distances.loc[min_dist:max_dist].copy(deep=True)
    
    means = distances.T.mean()
    stds  = distances.T.std()
    
    print('means:\n', means, '\n')
    print('stdev:\n', stds, '\n')
    
    for seed in distances.columns.values:
        distances[seed] -= means            ## mean-zero
        distances[seed] *= distances[seed]  ## this is taking the square
        distances[seed] /= stds             ## divide by stdev to get z-like
        
    most_normal = np.where(distances.sum() == min(distances.sum()))[0][0]
    sd = dict_of_excludes[exclude].columns[most_normal]
        
    print(f'most average seed for exclude {exclude} is {dict_of_excludes[exclude].columns[most_normal]} :')
    print(dict_of_excludes[exclude].loc[:,sd])
    
# =============================================================================
# %% plot the distributions of distances generated above
# =============================================================================
            
fig, ax = plt.subplots()

for exclude, distances in dict_of_excludes.items() :
    
    if distances.empty: continue
    
    ## plot the average number of sites
    ax.plot(distances.index.values, distances.T.mean(), label=exclude)
    
    ## plot the error bars
    ax.fill_between(distances.index.values, distances.T.mean()-distances.T.std(), 
                    distances.T.mean()+distances.T.std(), alpha=0.3)
            
ex = 4
sd = 748
## Plot the chosen structure
ax.scatter(dict_of_excludes[ex].index, dict_of_excludes[ex].loc[:,sd], label = 'chosen',
           zorder=3, marker='*', s=60, c = 'gold', edgecolors='k')

## pretty plot things. Top bound 265 for 116. 295 for 120; 170 for 106
ax.set(xlim=[0,5.5], ylim=[0,525], ylabel='Number of sites, per 800', xlabel='Distance to $Mg_{Al(2)}^\prime$, sites')
leg = ax.legend(title='min distance\n  between $Mg_{Al(2)}^\prime$')
leg.get_title().set_fontsize(12)
fig.tight_layout()

print(dict_of_excludes[ex].loc[:,sd])
            
    
    
    
    
    
    
            