#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:57:15 2020

USAGE: python3 hop_count.py metal stoich/rule temperature plane lammps_path variable

## example metal            : Na
## example stoich/rule      : 116           << works for beta and beta"
## example temperature      : 300
## example input_plane_name : 087           << name based on fractional z-coordinate
## example lammps_path      : ./Na116_5.lmp << needs to be in the same directory 
                                               as the hops and atoms folders
## example variable    : $E-$2 or w/e comes after 'hops' in the hop folder name

new example usage: python3 hop_count.py Na 116 300 012 ./Na116_5.lmp $E-$2

## edits 2020/07/23 : translated to python 3.6 and freud 2.2.0 with a working 
                      query and vectorization. Much faster! 
                      
## edits 2020/07/24 : eliminated the grid if loading a lammps file. The file
                      can be passed in the place of the folder - and needs to 
                      be in the same directory as the atoms and hops folders

@author: andreypoletaev
"""
# =============================================================================
# %% Initial imports
# =============================================================================

import numpy as np
import pandas as pd
import freud

from datetime import datetime as dt

from matplotlib import pyplot as plt

import hop_utils as hu
import crystal_utils as cu

import networkx as nx

import sys ## for sys.argv

## box size defaults to 1 just in case (fractional coordinates implied)
Lx, Ly, Lz = np.ones(3) 

# =============================================================================
# %% Parameters: 
# metal is the atom
# plane is 012,037,062,087 (string) for beta.
# in_folder is where the entire sweep of simulations is
# var is what changes in a sweep of simulations, e.g. 50x for E-field. 
# var is typically passed as command-line parameters to the shell file calling this
# =============================================================================

metal = sys.argv[1]
stoich = sys.argv[2]

## temperature [K] 
T1 = sys.argv[3] 

## 'Name' of an input plane, e.g. fractional z=0.875 means input_plane_name=087
input_plane_name = sys.argv[4]

## where the hop and atom folders are, typically '.' lammps file should be in the same folder
lammps_path = sys.argv[5]

## the name appended to the atom and hop folders 
var = sys.argv[6]

## TODO: implement better command-line argument passing with key=value
key_args = sys.argv[1:]

# =============================================================================
# %% testing input values from 2020/07/25 for beta
# ## available tests: Na122_4/Na100 750K, K116_5 300K
# =============================================================================

# metal = 'K' 
# stoich = '116_5' ## also available for testing: '122_4'
# T1 = '300'
# input_plane_name = '037'
# lammps_path = f'./testing/{metal}beta{stoich}.lmp'
# var = ''

# # testing folders
# gridfolder = './testing/grids/'
# hopfolder = './testing/hops/'
# atomfolder = f'./testing/trajectories/atoms-{metal}-{T1}/'

# =============================================================================
# %% testing input values from 2020/08/01 for beta-doubleprime
# ## available tests: Na_unsym_0 300K
# =============================================================================

# metal = 'Na' 
# stoich = 'unsym' 
# T1 = '300'
# input_plane_name = '016'
# lammps_path = f'./testing/{metal}_bdp_{stoich}_0.lmp'
# var = ''

# # testing folders
# gridfolder = './testing/grids/'
# hopfolder = './testing/hops/'
# atomfolder = f'./testing/trajectories/atoms-bdp-{metal}-{T1}/'

# =============================================================================
# %% read lammps file and get atoms from it, or assume that input to be a folder
# ## TODO: auto-identify BR sites
# =============================================================================

# Use fractional coordinates (if it works with the networks you have). 
# Real-space coords most likely work, but can give different network indices 
# from the fractional ones because of floating-point and wrapping on the boundaries
frac = False

lammps_passed = False
ointers = None
nearby_mg = None

try:
    # load the lammps file (omit nothing)
    _, _, cell, all_atoms = cu.read_lmp(lammps_path, fractional=frac)
    Lx, Ly, Lz = np.diag(cell) if not frac else np.ones(3)
    lammps_passed = True
    
    # make shorthand for Mg atoms and use it:
    # determine whether input lammps is doubleprime
    defects = all_atoms.query('type >= 4')
    defect_types = defects.atom.unique().tolist()
    dopants = defects.query('atom not in ["Al", "O"]')
    is_doubleprime = len(dopants) > 0
    
    # infer the z coordinates of conduction planes
    # This will yield no planes if the material is in fact Mg-doped beta.
    conduction_planes = cu.get_conduction_planes(all_atoms, metal, inexact=is_doubleprime)
    plane_names = ['{:03d}'.format(x) for x in ((conduction_planes/Lz+0.5) * 100).astype(int)]
    
    ## try again for Mg-doped beta
    if is_doubleprime and len(plane_names) < 1 :
        print('There are Mg defects, but planes do not look like beta".')
        conduction_planes = cu.get_conduction_planes(all_atoms, metal, inexact=False)
        plane_names = ['{:03d}'.format(x) for x in ((conduction_planes/Lz+0.5) * 100).astype(int)]
        is_doubleprime = False
    
    print(f'all conduction planes: {plane_names}, counting {input_plane_name}')
    
    # find out whether there are interstitial oxygens
    oxygen_types = all_atoms.query('atom == "O"').type.unique()
    
    # take out the oxygen interstitial defects if they are present
    # otherwise they will interfere with finding mobile-ion sites 
    if len(oxygen_types) > 1:
        ointers = all_atoms.query(f'type == {max(oxygen_types)}')
        print(f'Found {len(ointers)} oxygen interstitials.')
        
        all_atoms = all_atoms.query(f'type != {max(oxygen_types)}')
        
    elif is_doubleprime : print(f'beta"-alumina structure file with {len(dopants)} dopants, dopant types: {dopants.atom.unique()}')
    else : print(f'beta-alumina structure file with {len(dopants)} dopants, dopant types: {dopants.atom.unique()}')
    
    in_folder = '/'.join(lammps_path.split('/')[:-1]) + '/'
except:
    in_folder = lammps_path
    print('could not read lammps input file')
    max_ion_index = 20000 ## for non-consecutively numbered ions

## folder where the grid is - irrelevant if lammps file is passed
gridfolder = f'/home/groups/wchueh/apolet/old/grids/{stoich}/' ## for all M, 1.20 stoich

## folder where the by-atom trajectories are
atomfolder = f'{in_folder}/atoms{var}/' 

## folder where the hops get written
hopfolder = f'{in_folder}/hops{var}/'

# =============================================================================
# %% Find atom indices by plane
# TODO: connect to more generic files and integrate with crystal_utils
# (currently only works for standard beta files)
# TODO: switch to glob to find files with trajectories
# =============================================================================

# path to where the files with atoms are
filepath = atomfolder + metal + '{}.dat' ## one for every ion

if not lammps_passed:
    
    planes = np.zeros(4)
    ions = [[],[],[],[]]
    
    for i in np.arange(1, max_ion_index+1):
        filein = filepath.format(i)
        
        try:
            fi = open(filein,'r')
            line = fi.readline()
            arr = line[:-1].split(' ')
            z = int(float(arr[-1])*4.) ## TODO: modify for beta-doubleprime
            planes[z] += 1
            ions[z].append(i)
            fi.close()
            # print(f'ion {i} is in plane {z}')
        except:
            pass
            # print(f'no ions at index {i}')
            
    print('{} ions in folder'.format(sum([len(x) for x in ions])))
    ions_to_count = ions[int(float(input_plane_name)/100.*4.)]

## if the lammps file was read successfully, use query to find the right atom files
## TODO: adjust for beta-doubleprime with inexact z-coordinate matching
else: 
    
    mobile_ion_indices = dict()
    planes_by_name = dict()
    
    ## account for inexact z of mobile ions in conduction planes for doubleprime
    for p, pname in zip(conduction_planes, plane_names):
        thresh = 0.2 if not frac else 0.2/cell[2,2]
        mobile_ion_indices[pname] = all_atoms.query(f'{p - thresh} < z < {p + thresh} & atom == @metal').index.values
        planes_by_name[pname] = p
        
    ions_to_count = mobile_ion_indices[input_plane_name]

# =============================================================================
# %% get paths to site and atom lists. If they do not come from lammps, then 
# make the freud query object. This is a plane-specific block.
# =============================================================================

# path to file where hops will be written
hop_path = hopfolder  + metal + T1 + 'K_' + input_plane_name + 'plane.csv' 

# use the hop_utils method to make the freud Voronoi object and take from 
# that object the non-hardcoded number of cells
# here, the site_vor has already 'computed' and has polytopes
# here, site_pts are already wrapped
# (in the old version of counting hops, the make_voronoi() would return nn)
if not lammps_passed:
    grid_path = gridfolder + input_plane_name + 'grid.csv'
    box, site_vor, site_pts, numpolys, _ = hu.make_voronoi(grid_path)
    site_pts = np.asarray(site_pts)
else:
    ## shorthand variable for the z-coordinate of this plane
    plane = planes_by_name[input_plane_name]
    dz = np.mean(np.diff(conduction_planes)) 
    
    ## the box is for the query in the very end, irrespective of beta/beta"
    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
        
    ## immediately split by beta vs doubleprime. First is regular beta
    if not is_doubleprime and len(dopants) < 1:
        # get the mobile-ion sites for this plane
        # There should be a debugging print statement in get_mobile_ion_sites()
        site_pts = cu.get_mobile_ion_sites(all_atoms, plane, cell if not frac else np.eye(3), viz=False)
        num_sites = len(site_pts)
        print(f'Identified {num_sites} mobile-ion sites in plane {input_plane_name}')
        
        ## auto-get BR sites - this just yields a list of false for beta-doubleprime
        site_types = cu.auto_get_BR_sites(all_atoms, cell, site_pts, atoms_are_frac=frac)
        BR_sites = [i for i in range(len(site_types)) if site_types[i] == 'BR']
        
        # compose network of mobile-ion sites, calculate paths. Both beta and beta"
        # This can create and save a figure when viz=True. Pass cell if fractional was false
        _, edges, _ = cu.get_mid_oxygen_sites_freud(site_pts, cell if not frac else np.eye(3), viz=True)
        nxg = nx.from_edgelist(edges)
        path_lengths = cu.path_lengths(nxg)
        
        ## DEFECTS
        # find the interstitial defects in this plane: make an array and set z=0
        # the sites will be found below with the same query that counts hops
        try : 
            oi_defects = ointers.query(f'z == {plane}')
            oi_defect_pts = oi_defects[['x','y','z']].values
            # oi_defect_pts[:,-1] = 0
            
            oi_adjacent_sites, _ = cu.get_defect_adjacent_sites(cell, site_pts, oi_defect_pts)
            print(f'sites next to O_i in plane {input_plane_name}: ', oi_adjacent_sites)
            
            # calculate paths to interstitial-adjacent mobile-ion sites
            paths_to_oi = [min(path_lengths[oi_adjacent_sites,x]) for x in range(num_sites)]
        except : 
            # if no oxygens, then make very long paths
            print('(whether intentional or not, no O_i in this plane)')
            paths_to_oi = np.ones(num_sites).astype(int)*100
            
    ## Mg-doped beta
    elif not is_doubleprime and len(dopants) >= 1 :
        
        ## create a placeholder for the coordinates of the created defects
        defect_pts = dopants[['x', 'y', 'z']].values
        
        # get the mobile-ion sites for this plane
        # There should be a debugging print statement in get_mobile_ion_sites()
        site_pts = cu.get_mobile_ion_sites(all_atoms, plane, cell if not frac else np.eye(3), viz=False)
        num_sites = len(site_pts)
        print(f'Identified {num_sites} mobile-ion sites in plane {input_plane_name}')
        
        ## auto-get BR sites - this just yields a list of false for beta-doubleprime
        site_types = cu.auto_get_BR_sites(all_atoms, cell, site_pts, atoms_are_frac=frac)
        BR_sites = [i for i in range(len(site_types)) if site_types[i] == 'BR']
        
        ## get all the mid-oxygen sites in this plane
        ## create a proper networkx graph from site-edge list
        _, edges, _ = cu.get_mid_oxygen_sites_freud(site_pts, cell, viz=True)
        site_graph = nx.from_edgelist(edges)
        path_lengths = cu.path_lengths(site_graph)
        
        ## Mg DEFECTS
        ## find the Mg closest to each mobile-ion site in this plane
        e0, e1, d0, d1 = cu.get_nearest_points(site_pts, defect_pts, cell, num_nn=6)
        e0 = np.array(e0)[np.array(d0)<dz]
        e1 = np.array(e1)[np.array(d1)<dz]
        
        ## indices of mobile-ion sites
        s0 = [x[1] for x in e0]
        s1 = [x[1] for x in e1]
        mg0 = [x[0] for x in e0]
        mg1 = [x[0] for x in e1]
        
        ## Mg locations: ONLY for visualization
        # print('ordinal indices of relevant Mg defects:', mg0+mg1)
        nearby_mg = dopants.iloc[list(set(mg0+mg1))]
        
        # it will be more tidy for combining distances later to keep placeholder arrays
        if len(s0) > 0:
            paths_to_mg_0 = [min(path_lengths[s0, x]) for x in range(len(site_pts))]
        else: 
            paths_to_mg_0 = np.ones(len(site_pts))*len(site_pts)
        
        if len(s1) > 0:
            paths_to_mg_1 = [min(path_lengths[s1, x])+1 for x in range(len(site_pts))]
        else:
            paths_to_mg_1 = np.ones(len(site_pts))*len(site_pts)
        
        # combine path lengths to distance==1 and distance==0 sites using min()
        paths_to_oi = [min(paths_to_mg_0[i], paths_to_mg_1[i]) for i in range(len(site_pts))]
        
    ## doubleprime phase
    else:
        
        ## get mobile ion sites & paths, adjusted for above and below
        site_pts, _, _, path_lengths \
            = cu.get_sites_above_below(plane, all_atoms, cell if not frac else np.eye(3), 
                                       metal=metal, frac=frac, viz=True)
        num_sites = len(site_pts)
        
        ## select Mg sites next to this plane. Both sides are needed. 
        ## Furthermore, it is necessary to wrap around the boundary of the cell
        nearby_dop = dopants.query(f'{plane - dz} < z < {plane + dz}')
        
        ## accout for wrapping around the cell boundary
        if Lz*0.5 - abs(plane) < dz:
            nearby_dop2 = dopants.query(f'z < {-Lz + abs(plane) + dz} or z > {Lz - abs(plane) - dz}')
            # print(f'found wrapped Mg atoms: {len(nearby_mg2)}')
            nearby_dop = pd.concat([nearby_dop, nearby_dop2]).drop_duplicates()
        
        print(f'found total nearby dopant atoms: {len(nearby_dop)}')
        dopant_pts = nearby_dop[['x', 'y', 'z']].values
        
        ## find sites at 0-away and 1-away from defects
        edges_0, edges_1, d0, d1 = cu.get_nearest_points(site_pts, dopant_pts, cell if not frac else np.eye(3))
        sites_0 = [x[1] for x in edges_0]; mgs_0 = [x[0] for x in edges_0]; counts_0 = list()
        sites_1 = [x[1] for x in edges_1]; mgs_1 = [x[0] for x in edges_1]; counts_1 = list()
        
        ## calculate paths to Mg defects for all sites in this plane
        if len(sites_0) > 0:
            counts_0 = [[sites_0.count(x) for x in range(num_sites)].count(i) for i in range(6)]
            paths_to_mg0 = [min(path_lengths[sites_0, x]) for x in range(num_sites)]
        else: paths_to_mg0 = np.ones(num_sites)*num_sites
        if len(sites_1) > 0:
            counts_1 = [[sites_1.count(x) for x in range(num_sites)].count(i) for i in range(6)]
            paths_to_mg1 = [min(path_lengths[sites_1, x])+1 for x in range(num_sites)]
        else: paths_to_mg1 = np.ones(num_sites)*num_sites
            
        # combine path lengths to distance==1 and distance==0 sites taking min()
        paths_to_mg = [min(paths_to_mg0[i], paths_to_mg1[i]) for i in range(num_sites)]
        mg_counts = [(sites_1+sites_0).count(x) for x in range(num_sites)]
        counts = [mg_counts.count(x) for x in range(6)]

        ## print some distances
        print(f'Numbers of closest defects at 0: {counts_0}')
        print(f'Numbers of closest defects at 1: {counts_1}')
        print(f'Combined #s of closest defects : {counts}')
        print(f'distances at 0: {np.unique(np.round(d0,4))}')
        print(f'distances at 1: {np.unique(np.round(d1,4))}')
        
    ## VISUALIZATION
    ## save figure to know where the sites are in case there is confusion later
    if ointers is not None: plt.gca().scatter(oi_defects.x, oi_defects.y, c='r', s=75)
    elif nearby_mg is not None: 
        plt.gca().scatter(nearby_mg.x, nearby_mg.y, c='r', s=75)
    plt.gca().set(aspect=1 if not frac else cell[1,1]/cell[0,0], 
                  xlim=[-0.55*Lx, 0.55*Lx], ylim=[-0.55*Ly, 0.55*Ly])
    plt.gcf().tight_layout()
    plt.gcf().savefig(lammps_path[:-4]+f'_{T1}K_{input_plane_name}_{"frac" if frac else "real"}.png',dpi=300)
    
## initialize lists for the cases when they do not get used
if is_doubleprime:
    paths_to_oi = np.ones(num_sites) * num_sites
    BR_sites = list()
else:
    paths_to_mg = np.ones(num_sites) * num_sites
    
# remove the z coordinate for tracking hops
site_pts[:,-1] = 0 

# compose query parameters: one nearest neighbor appears enough
query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=False)

# create the freud query (freud 2.2.0)
que = freud.locality.AABBQuery(box, site_pts)    

# =============================================================================
# %% BIG LOOP: analyze all atoms one by one
# =============================================================================

# columns in the trajectory data file
data_cols = ['atom','time', 'x', 'y', 'z']

# flag for testing
do_fast = True

# list for by-atom dataframes
hops_list = list()

## set up data structures for all-atom statistics
## including partially complete stats to avoid double counting
try:
    predone_hops = pd.read_csv(hop_path)
    print('pre-done {} ions: '.format(len(predone_hops.ion.unique())), predone_hops.ion.unique())
    ions_to_count = set(ions_to_count).difference(set(predone_hops.ion.unique()))
    ions_to_count = sorted(ions_to_count)
    hops_list.append(predone_hops)
except IOError:
    print('starting afresh, no pre-completed ions.')
    total_hops = pd.DataFrame()

print(f'counting {len(ions_to_count)} ions:', ions_to_count)

dt_start = dt.now()

## loop for each ion
for z, ion in enumerate(ions_to_count) : 
    
    ## load file, set the time length of simulation, assume uniform time steps
    try:
        data = pd.read_csv(filepath.format(ion), sep = ' ', names=data_cols, header=None)
        time_cutoff = data.time.max()
        ts = (time_cutoff-data.time.iloc[0])/(len(data.time)-1)
        data.x *= Lx
        data.y *= Ly
        data.z *= Lz
    except:
        print(f'no file for ion {ion}, or something wrong with time stamps.')
        continue
    
    ## convert coordinates to the fractional freud points format 
    ## (freud is -L/2 to +L/2, lammps is 0 to L)
    ## TODO: save cell dims when parsing trajectories, & use real-space coords
    loci = data[['x','y','z']].values
    loci[:,0] -= 0.5*Lx
    loci[:,1] -= 0.5*Ly
    loci[:,2]  = 0
    num_loci = len(loci)
    
    dt0 = dt.now()

    ## find too many nearest neighbors, once for the entire trajectory 
    ## this is the old neighborList interface (freud 1.2.0)
    # nn.compute(box, loci, cell_centers)
    
    ## compute nearest neighbors
    result = que.query(loci, query_args)
    
    if do_fast:
    
        ## vectorize the output
        ## hop_mask checks for a change in the index of the nearest site, i.e. a hop
        arr_result = np.array(list(result))
        hop_mask = np.diff(arr_result[:,1]) != 0
        
        ## first cell
        first_cell = int(arr_result[0,1])
        
        ## if there are hops
        if sum(hop_mask) > 0:
            ## filter for the times at which hops occur
            fast_hop_times = data.time.iloc[1:].loc[hop_mask].values
            
            ## find the destinations of hops, plus the starting cell
            fast_cells = arr_result[1:,:][hop_mask,1].astype(int).tolist()
            fast_cells.insert(0,first_cell)
            
            ## find residence times, plus the last and first ones
            fast_resid_times = np.diff(data.time.iloc[1:].loc[hop_mask].values).tolist()
            fast_resid_times.append(data.time.iloc[-1] - max(fast_hop_times))
            fast_resid_times.insert(0, fast_hop_times[0])
            fast_resid_times = np.round(fast_resid_times,3)
            
            ## find out if hops get reversed
            rev_hop = [int(i==j) for i,j in zip(fast_cells[2:],fast_cells[:-2])]
            rev_hop.append(np.nan) ## we will never know if the last hop gets reversed
            
            ## distances to oxygens - if this is beta
            ox_rs = [paths_to_oi[x] for x in fast_cells]
            
            ## distances to mgs - if this is doubleprime
            mg_rs = [paths_to_mg[x] for x in fast_cells]
            
            ## BR sites - also if this is beta
            cell_is_BR = [x in BR_sites for x in fast_cells]
            
            ## counts of nearby defects - doubleprime
            nearby_mg = [mg_counts[x] if is_doubleprime else 0 for x in fast_cells]
            
            ## compose the hops dataframe
            if not is_doubleprime:
                hops = pd.DataFrame(data = {'ion':ion, 'time':fast_hop_times,
                    'old_cell':fast_cells[:-1], 'new_cell':fast_cells[1:],
                    'new_resid_time':fast_resid_times[1:], 'old_resid_time':fast_resid_times[:-1],
                    'new_ox_r':ox_rs[1:], 'old_ox_r':ox_rs[:-1],
                    'old_is_BR':cell_is_BR[:-1], 'rev_hop':rev_hop}) 
            else:
                hops = pd.DataFrame(data = {'ion':ion, 'time':fast_hop_times,
                    'old_cell':fast_cells[:-1], 'new_cell':fast_cells[1:],
                    'new_resid_time':fast_resid_times[1:], 'old_resid_time':fast_resid_times[:-1],
                    'new_mg_r':mg_rs[1:], 'old_mg_r':mg_rs[:-1],
                    'old_mg_count':nearby_mg[:-1], 'new_mg_count':nearby_mg[1:], 
                    'old_is_BR':cell_is_BR[:-1], 'rev_hop':rev_hop}) 
            
            ## write the final stats
            this_time = (dt.now()-dt_start).total_seconds()
            avg_onw = 100. - 100*hops.rev_hop.mean()
            print(f'ion {ion} ({z+1}/{len(ions_to_count)}): {sum(hop_mask)} hops, {avg_onw:.1f}% onward hops, {this_time:.2f} sec total', flush=True)
                   
        ## account for the possibility of no hops
        else:
            this_time = (dt.now()-dt_start).total_seconds()
            print(f'ion {ion}: NO HOPS! ({z+1}/{len(ions_to_count)}), {this_time:.2f} sec total', flush=True)
            
            first_is_BR = True if first_cell in BR_sites else False
            
            if not is_doubleprime :
                hops = pd.DataFrame(data = {'ion':[ion], 'time':[0.0], 
                    'old_cell':[first_cell], 'new_cell':[first_cell], 
                    'new_resid_time':[time_cutoff], 'old_resid_time':[time_cutoff],
                    'new_ox_r':[paths_to_oi[first_cell]], 'old_ox_r':[paths_to_oi[first_cell]],
                    'old_is_BR':[first_is_BR], 'rev_hop':[np.nan]})
            else:
                hops = pd.DataFrame(data = {'ion':[ion], 'time':[0.0], 
                    'old_cell':[first_cell], 'new_cell':[first_cell], 
                    'new_resid_time':[time_cutoff], 'old_resid_time':[time_cutoff],
                    'new_mg_r':[paths_to_mg[first_cell]], 'old_mg_r':[paths_to_mg[first_cell]],
                    'old_mg_count':[mg_counts[first_cell]], 'new_mg_count':[mg_counts[first_cell]],
                    'old_is_BR':[first_is_BR], 'rev_hop':[np.nan]})
        
    ## this is the old method that worked with freud 1.2.0 
    ## it is not vectorized b/c back then I could not bank on the neighborList
    ## working as advertised : it would return unpredictable #'s of neighbors
    else: 
        
        ## initialize a hop counter, hop times, residence times
        hop_times  = [] ## track timestamp for each hop event
        resid_time = [] ## track residence times between hops
        cell_resid = [] ## track cells between hops, matching residence times
        this_resid = 0  ## counter for residence time, starting with 1st step
        min_approach = []  ## track minimum distances for each visit to each cell
    
        ## find the correct nearest neighbor / cell at each timestep i, and detect hop
        for i in range(num_loci):
            ## get distances as the tiebreaker
            # dists = np.sqrt(nn.getRsq(i))
            subresult = [x for x in result if x[0] == i]
            dists = [x[2] for x in subresult]
            
            if len(subresult) != query_args['num_neighbors'] :
                print(f'at time {data.time[i]} found too few or too many neighbors')
            
            found = False
        
            ## comment out the old neighborList interface
            # for k, (d, j) in enumerate(zip(dists, nn.nlist.index_j[nn.nlist.index_i ==i])):
            for k, (tp, j, d) in enumerate(subresult) : 
                
                ## the closest point is somehow not in cell_centers
                if j >= numpolys : pass
                
                ## if the distance is the closest approach of the remaining distances at this time
                elif d == min(dists[k:]):
                    
                    d = np.round(d,6) ## save space in the CSV - does not work
                    
                    ## initialize event counters & detect hop
                    if i == 0:                  ## first cell must get counted
                        cell_resid.append(j)    ## first cell visited
                        min_approach.append(d)  ## first distance is minimum
                    ## hop event
                    elif cell_resid[-1] != j:           ## hop event
                        resid_time.append(np.round(this_resid,3))   ## append the last residence time
                        this_resid = ts                 ## reset residence counter, not to zero
                        cell_resid.append(j)            ## count new cell
                        min_approach.append(d)          ## first distance at new cell is the minimum
                        hop_times.append(data.time[i])  ## track the timestamp of every hop
                    ## no hop event
                    else :                      
                        # hop_events.append(0)    ## no hop
                        this_resid += ts        ## ion sits in the same cell, increase residence time
                        min_approach[-1] = min(min_approach[-1],d) ## re-calc approach distance
                    found = True
                    break
                ## distance is not minimum of remaining NN distances, weird
                else : 
                    print(f'NN distance not minimum at {data.time[i]} ps, weird :/')
                    pass
            
            ## shout out if location missing
            if not found :
                print(f'could not find site for ion {ion} at time {data.time[i]} ps')
            
            ## shout out a completed ion
            if (10*i) % num_loci == 0:
                print('ion {} {}% done, {} hops, {:.2f} sec'.format(ion, 
                            100*i/num_loci, len(cell_resid[1:]), (dt.now()-dt0).total_seconds()), flush=True)

        ## The above does not count the residence time at the last site, 
        ## irrespective of whether the last timestep is a hop or not
        ## So that last residence time needs to get appended manually
        resid_time.append(np.round(this_resid,3))
    
        ## count to-from cells for each hop event
        if len(cell_resid) > 1:
            hop_from = cell_resid[:-1]
            hop_to = cell_resid[1:]
    
        ## count back-hops (that are returns) & rev hops (that will reverse)
        if len(cell_resid) > 1:
            # back_hop = [int(i==j) for i,j in zip(hop_to[1:],hop_from[:-1])]
            # back_hop.insert(0,0) ## first hop cannot be a back-hop
            
            rev_hop = [int(i==j) for i,j in zip(hop_to[1:],hop_from[:-1])]
            rev_hop.append(np.nan) ## we will never know if the last hop gets reversed
    
        ## put together a hopping dataframe
        ## note: this ignores the starting cell
        ## maybe the finish cell should be tossed as well, using [:-1]
        if len(cell_resid) > 1:
            hops = pd.DataFrame(data = {'time':hop_times,'old_cell':hop_from,
                                        'ion':ion,'new_cell':hop_to, 
                                        'new_resid_time':resid_time[1:], 'rev_hop':rev_hop,
                                        'new_min_dist':np.round(min_approach[1:],7)})
        
            ## final stats
            print('ion {}: {} hops, {:.1f}% onward hops (ion {}/{}), {} sec total'.format(ion, 
                       len(cell_resid)-1, 100.-100*hops.rev_hop.mean(), 
                       z+1, len(ions_to_count), round((dt.now()-dt_start).total_seconds(),2)), flush=True)
        
        ## Account for the possibility of no hops
        else: 
            print('ion {}: NO HOPS! ({}/{}), {} sec total'.format(ion, z+1, 
                       len(ions_to_count), round((dt.now()-dt_start).total_seconds(),2)), flush=True)
            hops = pd.DataFrame(data = {'time':0.0,'old_cell':cell_resid,
                                        'ion':ion,'new_cell':cell_resid,
                                        'new_resid_time':time_cutoff, 'rev_hop':np.nan,
                                        'new_min_dist':np.round(min_approach[-1],7)}) 
    
    ## add this ion's stats to total stats
    hops_list.append(hops)
    
    ## save results after every 10th ion, in case the running gets interrupted
    if z % 10 == 0: 
        total_hops = pd.concat(hops_list)
        total_hops.to_csv(hop_path, index=False)

# =============================================================================
# save results in case the previous cell was interrupted
# =============================================================================
total_hops = pd.concat(hops_list)
total_hops.to_csv(hop_path, index=False)
print('completed ions saved', total_hops.ion.unique())

# =============================================================================
# load ions to check
# =============================================================================
total_hops = pd.read_csv(hop_path)
print('completed ions recovered', total_hops.ion.unique())
    




  
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    