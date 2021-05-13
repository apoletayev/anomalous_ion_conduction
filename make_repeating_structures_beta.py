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

import sys, os, random
if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import pandas as pd
import numpy  as np
import networkx as nx
import crystal_utils as cu

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

## This is the number of interstitials to add to every plane
## (the code enables varying the number for every plane as well)
num_ois_per_plane = 10

start = dt.now()

## read the input .CONFIG file & add symmetry elements
phase, intro_line, cell, atoms = cu.read_poly(prod_file, fractional=False)
# atoms = cu.add_symmetry_beta(atoms, cell, mobile=metal, frac=False)

## find all conduction planes in the input file
planes = cu.get_conduction_planes(atoms, metal)

## generate networks and path lengths for every plane, only once 
mid_ox_dict = dict()
path_length_dict = dict()
mobile_site_dict = dict()

for pl in planes : 
    these_mobile_sites = cu.get_mobile_ion_sites(atoms, pl, cell)
    these_mid_oxs, these_edges, _ = cu.get_mid_oxygen_sites_freud(these_mobile_sites, cell, viz=False)
    path_length_dict[pl] = cu.path_lengths(nx.from_edgelist(these_edges))
    mid_ox_dict[pl] = these_mid_oxs
    mobile_site_dict[pl] = these_mobile_sites
    print(f'Made networks in plane z={pl:.4f}, total {(dt.now()-start).total_seconds():4.1f} sec.')
    
dict_of_excludes = dict()
print(f'Setup took {(dt.now()-start).total_seconds():4.1f} sec.')

# =============================================================================
# %% Generate a bunch of random structures and count their sites
# =============================================================================

start = dt.now()

for exclude in [1,4] :

    ## dictionary of histograms of paths to Oi-closest sites
    hists_dict = dict()
    
    ## iterate over random seeds
    for seed in range(100) :
        
        ## reset seed
        random.seed(seed)
        
        ## start timer for sanity
        seed_start = dt.now()
    
        ## initialize data structures
        all_paths_to_oi = list() 
        broken=False
        
        ## for every plane, find mid-oxygen sites
        for pl in planes:
            
            ## if a previous plane did not work, short-circuit
            if broken: break
            
            mobile_sites = mobile_site_dict[pl]
            mid_oxs = mid_ox_dict[pl]
            path_lengths = path_length_dict[pl]
        
            ## Pick mid-oxygen sites quasi-randomly, independent of coordinates
            picked, _, _ = cu.generate_mid_oxygens(mid_oxs, num_ois_per_plane, exclude)
            # picked, _, _ = cu.generate_mid_oxygens_packed(mid_oxs, num_ois_per_plane, exclude)
            
            if len(picked) < num_ois_per_plane : 
                broken=True
                break
    
            ## save the sites as oxygen sites for later
            metal_sites_next_to_oi = np.array(sorted([x for x in cu.flatten(list(picked))]))
        
            ## measure all path lengths to the oxygens; this yields a list
            all_paths_to_oi.append([min(path_lengths[metal_sites_next_to_oi,x]) for x in range(len(mobile_sites))])        
            
        if not broken: 
            ## remove nestinging in the path lengths to oxygen sites
            all_paths_to_oi = np.array([x for x in cu.flatten(all_paths_to_oi)])
            
            ## save the histogram
            hists_dict[seed] = np.histogram(all_paths_to_oi, np.arange(-0.5,0.5+max(map(max,path_lengths))))[0]
            
            ## show a sign of life
            print(f'({exclude:2d},{seed:2d}) took {(dt.now()-seed_start).total_seconds():4.1f} sec, \
                  total {(dt.now()-start).total_seconds():4.1f} sec')
        else:
            print(f'({exclude:2d},{seed:2d}) took {(dt.now()-seed_start).total_seconds():4.1f} sec, was broken')
        
    ## dataframe with all distances by seed
    dict_of_excludes[exclude] = pd.DataFrame(data = hists_dict)
    
# =============================================================================
# %% compute z-scores to find the most-average structure 
# =============================================================================

min_dist=1
max_dist=10

for exclude, distances in dict_of_excludes.items():
    
    if distances.empty: continue
    
    distances = distances.loc[min_dist:max_dist].copy(deep=True)
    
    means = distances.T.mean()
    stds  = distances.T.std()
    
    print('\nmeans:\n', means)
    print('\nstds:\n', stds)
    
    for seed in distances.columns.values:
        distances[seed] -= means            ## mean-zero
        distances[seed] *= distances[seed]  ## this is taking the square
        distances[seed] /= stds             ## divide by stdev to get z-like
        
    most_normal = np.where(distances.sum() == min(distances.sum()))
        
    print(f'most average seed for exclude {exclude}: {most_normal[0]}\n=====')
    
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
sd = 15

## Plot the chosen structure
ax.scatter(dict_of_excludes[ex].index, dict_of_excludes[ex].loc[:,sd], label = 'chosen',
           zorder=3, marker='o', s=16, c = 'yellow', edgecolors='k')

## pretty plot things. Top bound 265 for 116. 295 for 120; 170 for 106
ax.set(xlim=[0,10], ylim=[0,295], ylabel='Number of sites, per 800', xlabel='Distance to $O_i$, sites')
leg = ax.legend(title='min distance\n  between $O_i$')
leg.get_title().set_fontsize(12)
fig.tight_layout()
            
    
    
    
    
    
    
            