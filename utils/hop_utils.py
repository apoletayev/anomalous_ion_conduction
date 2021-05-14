#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:10:09 2019

@author: andreypoletaev
"""

import numpy as np
import pandas as pd
import freud
from scipy.spatial import Voronoi

from matplotlib import pyplot as plt
import matplotlib as mpl
from colorsys import rgb_to_hls, hls_to_rgb

from scipy.signal import butter, filtfilt
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit as cf
from scipy.special import erf, gamma
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from timeit import default_timer as timer

from datetime import datetime as dt

import crystal_utils as cu
import networkx as nx

from itertools import chain, repeat

from os import path
from glob import glob

from re import split

from batlow import cm_data as batlow_cm

kb = 8.617e-05 ## [eV/Kelvin]

# =============================================================================
# %% cosmetic things
# =============================================================================

## colors for mobile ions
metal_colors = {'Na':'tab:orange', 'K':'#7d02d4', 'Ag':'tab:red',
                'Li':'tab:green'}

## perceptually uniform color map, Crameri, F. Scientific Colour Maps, http://www.fabiocrameri.ch/colourmaps (2020).
## (and the Nat Comms paper, doi.org/10.1038/s41467-020-19160-7 )
# batlow_map = LinearSegmentedColormap.from_list('batlow', batlow_cm)
batlow_cdict = {'red' : np.array(batlow_cm)[:,0], 
                'green' : np.array(batlow_cm)[:,1],
                'blue' : np.array(batlow_cm)[:,2]}

batlow_hls = [rgb_to_hls(*i) for i in batlow_cm]

def batlow_lightness_scaled(min_l=batlow_hls[0][1], max_l=batlow_hls[-1][1]):
    ''' Adjusts lightness on the otherwise perceptually uniform colormap.
        Returns in rgb format. '''
    
    linear_lightnesses = np.linspace(batlow_hls[0][1], batlow_hls[-1][1], 256)
    
    nonlinearity = [ i[1]-j for i, j in zip(batlow_hls, linear_lightnesses)]
    
    scaling = abs(max_l - min_l) / abs(batlow_hls[-1][1] - batlow_hls[0][1])
    
    new_linear_lightnesses = np.linspace(min_l, max_l, 256)
    
    new_lightnesses = [scaling*n + nll for nll, n in zip(new_linear_lightnesses, nonlinearity)]
    
    return [hls_to_rgb(b[0], l, b[2]) for b,l in zip(batlow_hls, new_lightnesses)]
                
## dictionary of colors to make a LinearSegmentedColormap 
## that is like coolwarm but with a darker middle
cdict = {'blue':[[0., 1., 1.], [0.5,0.6,0.6], [1., 0., 0.]],
         'green':[[0., 0., 0.],[0.5,0.6,0.6], [1., 0., 0.]],
         'red':[[0., 0., 0.],  [0.5,0.6,0.6], [1., 1., 1.]] }

zs = ['z_all', '012', '037', '062', '087']
single_zs = ['012', '037', '062', '087']

dims = {'x':0, 'y':1, 'z':3, 'dx':0, 'dy':1, 'dz':3}

# =============================================================================
# %% a list flattening function for lists of strings (filenames)
# ## flatten returns an iterator (usually sufficient), 
# ## flattened makes it into a proper list
# =============================================================================

flatten = lambda l: chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l)

def flattened(nested_list_input):
    
    flat_list_output = list()
    for x in flatten(nested_list_input) : flat_list_output.append(x)
    
    return flat_list_output

# =============================================================================
# %% aggregation function for edges of graphs
# ## this is used in plots of correlations, the col is typically 'dt' 
# =============================================================================

agg_edges_time = lambda df, col: df.groupby(df[col].apply(lambda x: round(x, 2))).count()

# =============================================================================
# %% running-average function for smoothing curves - especially RDF
# =============================================================================

def running_mean(x,N) :
    cumsum = np.cumsum(np.insert(x,0,0))
    return (cumsum[N:] - cumsum[:-N])/N

# =============================================================================
# %% define helper functions
# =============================================================================

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

def rot(angle):
    '''rotation matrix'''
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

def in_bounds(pt, xymin, xymax):
    ''' check if point is between (xy)_min and (xy)_max
        2020/06/13 : added left side equality 
        This is a general method for any number of dimensions '''
    
    return (sum(pt >= xymin) == len(pt)) & (sum(pt <= xymax) == len(pt))

# =============================================================================
# %% visualization function from freud. This method started here:
# ## freud.readthedocs.io/en/v1.2.0/examples/module_intros/Voronoi-Voronoi.html
# =============================================================================

def draw_voronoi(box, points, cells, nlist=None, color_by_sides=False, ax=None,
                 draw_points=False, draw_box=False, single_poly_color=None,
                 cell_numbers=None, cell_text=None, color_by_property=None,
                 draw_nns=False, skip_polys=False, property_clim=[0, 1],
                 alpha=0.4, cmap='coolwarm', draw_colorbar=False):
    ''' This method started out here:
    freud.readthedocs.io/en/v1.2.0/examples/module_intros/Voronoi-Voronoi.html
    
    AP added simple flags for: draw_pts, draw_box, coloring by a property,
    writing cell numbers next to the plotted polygons. 
        
    Distinction between 'points' and 'cells': points are ALL the points (centers 
    of Voronoi polygons) in the tessellation, while cells are the polygons 
    (polytopes in freud parlance) that are to be visualized. Therefore, 
    len(cells) must be <= len(points), otherwise an error will be thrown.
    
    Coloring by a property: pass an iterable, color_by_property, with indices 
    matching the cells to be plotted, and min/max limits on it via property_clim
    Cell numbers: pass an iterable, cell_numbers, of numbers that matches the 
    length of polygons (input points) to be plotted AND the indices of the cells
    (among points) that are getting plotted. Custom labels with cell_text.
    '''
    ## AP adds specifying axes in which to make the plot
    if ax is None: ax = plt.gca()

    # Draw Voronoi cells
    patches = [plt.Polygon(cell[:, :2]) for cell in cells]
    patch_collection = mpl.collections.PatchCollection(patches, alpha=alpha,
                                                       # edgecolors='black', ## AP took out
                                                       facecolors=single_poly_color)

    if single_poly_color is not None:  ## No color map needed
        colors = [1 for cell in cells]
        bounds = [1, 1]
        patch_collection.set_edgecolor('black')
    elif color_by_property is None:  ## need to make a color map
        if color_by_sides:
            colors = [len(cell) for cell in cells]  ## original said voro.polytopes for the full graph, 
            ## this allows plotting a specific subset of cells
        else:  ## choose random colors for all polygons
            colors = np.random.permutation(np.arange(len(patches)))
            patch_collection.set_edgecolor('black')
        #        cmap = plt.cm.Set1 ## AP took out
        cmap = plt.cm.get_cmap('Set1', np.unique(colors).size)  ## this essentially ranks all the cells without checking absolute differences
        bounds = np.array(range(min(colors), max(colors) + 2))
        ## more finagling can be done here to normalize color map if need be
    elif color_by_property is not None:  ## color by property
        if type(color_by_property) == pd.core.series.Series:
            colors = [color_by_property.iloc[i] for i, cell in enumerate(cells)]  ## normalized below
        else : 
            colors = [color_by_property[i] for i, cell in enumerate(cells)]
        bounds = property_clim
        cmap = plt.cm.get_cmap(cmap)  ## assuming 256 colors resolution

    patch_collection.set_array(np.array(colors))
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0], bounds[-1])

    ## option to skip the polytopes and only do other stuff
    if not skip_polys:
        ax.add_collection(patch_collection)

    # Draw points
    if draw_points:
        pt_colors = np.random.permutation(np.arange(len(points)))  ## AP
        plt.scatter(points[:, 0], points[:, 1], c=pt_colors, s=6)  ## AP change to pt_colors

    ## AP: Write the numbers of polygons, given #cells ≤ #points
    ## 2020/07/09: comment out the old, add the option for custom text
    if cell_numbers is not None:
        # for c, cn in zip(cells, cell_numbers):
        #     ax.text(points[cn, 0], points[cn, 1], cn, fontsize=8)
        for i, cn in enumerate(cell_numbers):
            text = f'({cn},\n{cell_text[i]})' if cell_text is not None else cn
            ax.text(points[cn, 0], points[cn, 1], text, fontsize=8)

    plt.xlim((-box.Lx / 2, box.Lx / 2))
    plt.ylim((-box.Ly / 2, box.Ly / 2))

    ## Set equal aspect and draw the box
    if draw_box:  ## added AP
        #        ax.set_aspect('equal', 'datalim') ## commented out AP
        box_patch = plt.Rectangle([-box.Lx / 2, -box.Ly / 2], box.Lx, box.Ly, alpha=1, fill=None)
        ax.add_patch(box_patch)

    ## Draw nearest-neighbor lines: this is freud 1.2.0 back-compatibility
    ## For freud 2.2.0, use the below with flag draw_neighbors
    if nlist is not None:
        bonds = np.asarray([points[j] - points[i] for i, j in zip(nlist.index_i, nlist.index_j)])
        box.wrap(bonds)
        line_data = np.asarray([[points[nlist.index_i[i]],
                                 points[nlist.index_i[i]] + bonds[i]] for i in range(len(nlist.index_i))])
        line_data = line_data[:, :, :2]
        line_collection = mpl.collections.LineCollection(line_data, alpha=0.3)
        ax.add_collection(line_collection)

    ## connect nearest neighbors, freud 2.2.0
    if draw_nns > 0:
        que = freud.locality.AABBQuery(box, points)
        query_args = dict(mode='nearest', num_neighbors=draw_nns, exclude_ii=True)
        result = list(que.query(points, query_args))
        bond_vectors = np.asarray([points[x[1]] - points[x[0]] for x in result])
        bond_vectors = box.wrap(bond_vectors)
        line_data = [[points[result[i][0]], points[result[i][0]] + bond_vectors[i]] \
                     for i in range(len(result))]
        line_data = np.asarray(line_data)[:, :, :2]  ## planarize
        line_collection = mpl.collections.LineCollection(line_data, alpha=0.3)
        ax.add_collection(line_collection)

    # Show colorbar for number of sides
    if color_by_sides or draw_colorbar:
        cb = plt.colorbar(patch_collection, ax=ax, ticks=bounds, boundaries=bounds)
        cb.set_ticks(cb.formatter.locs + 0.5)
        cb.set_ticklabels((cb.formatter.locs - 0.5).astype('int'))
        cb.set_label("Number of sides", fontsize=12)

# =============================================================================
# %% generating a lattice of repeating points for making the Voronoi grid
# ## Deprecated: the grid is generated from the crystal structure file
# =============================================================================
def gen_grid(pt0, a1, a2, angle=0, angle2=np.pi/6, xymin=np.array([-0.05,-0.1]), xymax=np.array([1.05,1.15])):
    ''' generate array of points within bounds '''
    
    ## initiate list of points    
    xs = [pt0[0]]
    ys = [pt0[1]]
    
    va1 = np.dot(rot(angle), np.array([a1,0]))
    va2 = np.dot(np.dot(rot(angle),rot(angle2)), np.array([a2,0]))
        
    numa1 = (xymax-xymin) / va1
    numa2 = (xymax-xymin) / va2
    
    zeroa1 = (pt0-xymin) / va1
    zeroa2 = (pt0-xymin) / va2
        
    for a1i in np.round(np.arange(-max(zeroa1)*5,max(numa1-zeroa1)*5)) : ## x direction
        for a2i in np.round(np.arange(-max(zeroa2)*5,max(numa2-zeroa2)*5)) : ## y direction
            pt = pt0 + a1i * va1 + a2i * va2
            if in_bounds(pt,xymin,xymax):
                xs.append(pt[0])
                ys.append(pt[1])
                
    return xs, ys

# =============================================================================
# %% check neighbors with arbitrary connection indices
# ## this is only relevant for older versions of freud. 
# ## DEPRECATED; was used with freud 1.2.0 back in 2019
# =============================================================================
    
def are_neighbors(cell1s, cell2s, index_i, index_j):
    '''
    return true if cell2s are nearest-neighbors of cell1s, 
    using indices index_i and index_j
    Note that indices are arbitrary and could be 2-nearest, 1-nearest, or else
    '''
    assert len(cell2s) == len(cell1s), 'cell1s and cell2s must be same length'
    
    return [cell2s[k] in index_j[np.where(index_i==cell1s[k])] for k in range(len(cell1s)) ]

# =============================================================================
# %% make dictionaries for distances to oxygens and for path lengths
# =============================================================================
    
def edge_distances_to_oi(folder='.', zs=['z_all', '012', '037', '062', '087'],max_r=4):
    ''' returns dict of (max_r+1) x (max_r+1) matrices with counts of edges 
        aggregated by their distances to Oi sites. max_r must be passed as input
        max_r can be calculated from the "old_ox_r" or "new_ox_r" columns
        
        2020/06/16: saving time, save a set of distances for every folder,
        and first try to look it up and load it'''
        
    distances = dict()

    for zz in zs:
        try: distances[zz] = np.loadtxt(folder+'/edge dists {}.csv'.format(zz)).astype(int)
        except: 
            ## calculate from paths and Oi distances
            try:
                dists = np.zeros((max_r+1,max_r+1))
                oi_sites = np.loadtxt(folder+'/oxygen_cells_{}.csv'.format(zz)).astype(int)
                paths = np.loadtxt(folder+'/paths{}.csv'.format(zz),delimiter=',').astype(int)
#                print('loaded site-to-site paths')
                for edge in np.argwhere(paths==1): ## count all 1-NN paths
                    dists[min(paths[oi_sites,edge[0]]),min(paths[oi_sites,edge[1]])] += 0.5 if edge[0] != edge[1] else 1
                
#                print('processed all edges')
                ## remove zeros as this will be divided by
                for i in range(len(dists)):
                    for j in range(len(dists[0])):
                        if dists[i,j] == 0: dists[i,j] = 1
                
#                print('removed zeros')
                ## assign dictionary
                distances[zz] = dists
                
                ## save
                np.savetxt(folder+'/edge dists {}.csv'.format(zz),distances[zz],fmt='%d',delimiter=',')
            except: print(f'missing paths or Oi cells at z = {zz}')
        
    return distances

def sites_by_distance(folder='.', zs=['z_all', '012', '037', '062', '087']):
    ''' returns dict of lists w/indices of sites by their network distance 
        from the nearest Oi, from within the specific folder.
        This is usually where sites_by_r[plane][radius] calls come from ''' 
    sites = dict()
    
    for zz in zs:
        try:
            oi_sites = np.loadtxt(folder+'/oxygen_cells_{}.csv'.format(zz)).astype(int)
            paths = np.loadtxt(folder+'/paths{}.csv'.format(zz),delimiter=',').astype(int)
            path_lengths = sorted(set(flatten(paths)))
            _ = path_lengths.pop() ## take out longest 
            sites[zz] = [[] for p in range(max(path_lengths)+2)] ## could in principle throw an error
            for s in range(len(paths)) : ## each path is a square matrix
                sites[zz][min(paths[oi_sites,s])].append(s)
            for r in range(len(sites[zz])): ## prune each from end
                if not sites[zz][-1] : sites[zz] = sites[zz][:-1]
        except: print(f'something missing for z = {zz}')
        
    return sites
    

def BR_sites(folder='.', zs=['z_all', '012', '037', '062', '087']) :
    ''' retrieves (from pre-saved files) the 1D arrays with network indices of 
        Beevers-Ross sites for beta-aluminas. All this is only
        in reference to a pre-defined "grid" that defines the network.
        AP moved from hop_utils 2020/06/23 '''
    BR_dict = dict()
    
    for zz in zs:
        try:
            BR_dict[zz] = np.loadtxt(folder+'/sites BR {}.csv'.format(zz),delimiter=',')
        except: print('missing BR sites at z = {zz} in folder {folder}')
            
    return BR_dict

def site_paths(folder = '.', zs=['z_all', '012', '037', '062', '087']):
    ''' retrieves number_of_polygons x number_of_polygons arrays with network path lengths
        between mobile-ion sites from pre-saved files. All this is only in 
        reference to a pre-defined "grid" that defines the network. 
        AP moved from hop_utils 2020/06/23 '''
    paths = dict()
    
    for zz in zs:
        try:
            paths[zz] = np.loadtxt(folder+'/paths{}.csv'.format(zz),delimiter=',').astype(int)
        except: print(f'missing paths at z = {zz}')
    
    return paths

def o_sites(folder = '.', zs=['z_all', '012', '037', '062', '087']):
    ''' retrieves (from pre-saved files) the 1D arrays with network indices of sites
        that have an edge that is occupied by an O_interstitial. All this is only
        in reference to a pre-defined "grid" that defines the network.
        AP moved from hop_utils 2020/06/23 '''
    paths = dict()
    
    for zz in zs:
        try:
            paths[zz] = np.loadtxt(folder+'/oxygen_cells_{}.csv'.format(zz),delimiter=',').astype(int)
        except: print(f'missing O sites z = {zz}')
    
    return paths

# =============================================================================
# %% load a list of hops for a plane nicely
# ## update for adding reverse column if it is not already there
# ## (it was recorded by ion and in simulation-chronological order)
# ## 2020/04/07: adding oxygen path input as 'old_ox_r', 'new_ox_r'
# ## 2020/06/13: adding the do_update flag
# =============================================================================
    
def load_plane(path, numpolys=200, numplanes=1, verbose=True, do_update=True,
               oxygen_path=None, paths_path=None, BR=[]):
    
    total_hops = pd.read_csv(path); save_update = False
    zz = which_one_in(zs,path)
    
    if 'new_mg_r' in total_hops.columns: 
        do_update=False
        save_update=False

    ## check reverse column and update if needed
    if do_update and 'rev_hop' not in total_hops.columns:
        
        print('\nupdating reverse hops column...')
    
        total_hops = total_hops.assign(rev_hop = np.zeros(len(total_hops)))
        num_ions = len(total_hops.ion.unique())
    
        for ii, ion in enumerate(total_hops.ion.unique()):
        
            one_ion_hops = total_hops.loc[total_hops.ion == ion]
            
            ## account for non-hops
            if len(one_ion_hops) > 1:
                rev = list((one_ion_hops.new_cell[1:].values == one_ion_hops.old_cell[:-1].values))
                rev.append(np.nan) ## last hop is unknown
            else : rev = [np.nan]
            
            total_hops.rev_hop.loc[total_hops.ion == ion] = rev
        
            if ii % 25 == 0: print(f'{ii} of {num_ions} ions done')
            
        save_update = True; print('... done and onwards.')
    elif verbose: print('\nupdate complete: reverse hopping.')
    
    ## check residence time of previous hop for correlations
    if do_update and 'old_resid_time' not in total_hops.columns :
    
        print('updating old residence time column...')
    
        total_hops = total_hops.assign(old_resid_time = np.zeros(len(total_hops)))
        num_ions = len(total_hops.ion.unique())
    
        for ii, ion in enumerate(total_hops.ion.unique()):
        
            one_ion_hops = total_hops.loc[total_hops.ion == ion]
            
            ## account for non-hops
            if len(one_ion_hops) > 1:
                old = list(one_ion_hops.new_resid_time[:-1])
                old.insert(0,one_ion_hops.time.iloc[0]) ## 1st hop unknown - and that is OK
            else : old = [np.nan]
            
            total_hops.old_resid_time.loc[total_hops.ion == ion] = old
        
            if ii % 25 == 0: print(f'{ii} of {num_ions} ions done')
            
        save_update = True; print('... done and onwards.')
    elif verbose: print('update complete: old residence time.')
        
    ## add columns on the distance to nearest oxygen to all planes
    while do_update and ('old_ox_r' not in total_hops.columns or 'new_ox_r' not in total_hops.columns) :
        if oxygen_path is None or paths_path is None: 
            print('distances to Oi missing; add oxygen_path=... and paths_path=... to update')
            break ## out of the while loop
        elif oxygen_path == 'no interstitials' :
            total_hops['new_ox_r'] = 100
            total_hops['old_ox_r'] = 100
            break
            
        print('updating distances to oxygens...')
        oi_sites = np.loadtxt(oxygen_path).astype(int)
        paths = np.loadtxt(paths_path,delimiter=',').astype(int)
        
        ## add columns
        total_hops['new_ox_r'] = total_hops.new_cell.apply(lambda x: min(paths[oi_sites,x]))
        total_hops['old_ox_r'] = total_hops.old_cell.apply(lambda x: min(paths[oi_sites,x]))
        
        ## save & update
        save_update = True; print('... done and onwards.')
    if 'old_ox_r' in total_hops.columns and 'new_ox_r' in total_hops.columns and verbose :
        if not save_update: print('update complete: distances to oxygens.')
        
    ## add the BR column here - if beta
    if 'old_is_BR' not in total_hops.columns and do_update : 
        print('updating BR site column...')
        total_hops['old_is_BR'] = total_hops.old_cell.isin(BR[zz])
        ## save & update
        save_update = True; print('... done and onwards.')
    elif verbose: print('update complete: BR sites (hop origin).')
        
    if save_update: ## save all updates at once
        print('saving updated hops...')
        total_hops.to_csv(path,index=False)
        
    ## proceed to actually load stuff
    all_residences = total_hops.groupby('new_cell').mean().new_resid_time
    # new_cells = total_hops.new_cell.unique()
    # old_cells = total_hops.old_cell.unique()
    
    non_hops = total_hops.loc[total_hops.new_cell == total_hops.old_cell]
    # empties = set(range(numpolys*numplanes))-set(new_cells)-set(old_cells)
        
    empties = set(range(numpolys*numplanes))-set(list(all_residences.index))
    non_hop_sites = list(non_hops.new_cell.unique())

    ## take out the placeholders for the ions that do not hop
    ## those were recorded as one hop with new_cell == old_cell
    total_hops = total_hops.loc[total_hops.new_cell != total_hops.old_cell]

    if verbose:
        print('\n{} ions hopped {} times, {} ions stayed put'.format(
            len(total_hops.ion.unique()),len(total_hops), len(non_hops)))
        if 'old_is_BR' in total_hops.columns:
            print('{} hops from BR sites, {} from aBR sites'.format(
                    len(total_hops.query('old_is_BR == True')),
                    len(total_hops.query('old_is_BR == False'))))
        print('{} total onwards hops, {:.1f}% hops reversed'.format(
            total_hops.rev_hop.loc[total_hops.rev_hop==False].size,
            total_hops.rev_hop.loc[np.isnan(total_hops.rev_hop)==False].mean()*100.))
        print(f'{numpolys*numplanes} sites, {len(empties)} remained empty')
    else :
        print('...plane loaded')
    
    return total_hops, all_residences, non_hops, empties, non_hop_sites

# =============================================================================
# %% combine planes after updating
# ## skipping the graphics for a moment
# =============================================================================

def combine_planes3(plane_paths, zs, numpolys=200, verbose=False):
    
    combined_hops = pd.DataFrame()
    
    ## sort the inputs by z, while keeping them together
    all_inputs = list(zip(plane_paths, zs))
    all_inputs = sorted(all_inputs, key = lambda x: x[1])
    
    if verbose: print(f'combining {len(all_inputs)} planes')
    
    for i, (plane, z) in enumerate(all_inputs):
        
        th, _, nh, _, _ = load_plane(plane, numpolys=numpolys, verbose=True)
        these_hops = pd.concat([th,nh],ignore_index=True)
        
        these_hops.new_cell += i * numpolys
        these_hops.old_cell += i * numpolys
        
        these_sites = set(these_hops.new_cell.unique()).union(set(these_hops.old_cell.unique()))
        
        combined_hops = pd.concat([combined_hops, these_hops],ignore_index=True)
        
        if verbose:
            # print(f'ions at z={z:03d} :', these_hops.ion.unique())
            print('sites from {} to {}, {} total\n'.format(min(these_sites),max(these_sites),len(these_sites)))
        
    return combined_hops

# =============================================================================
# %% low-pass filter
# =============================================================================

def lopass(signal, cutoff, sampling_freq, order = 5):
    
    nyquist = sampling_freq / 2
    b, a = butter(order, cutoff/nyquist)
    
    if not np.all(np.abs(np.roots(a)) < 1):
        raise ValueError('Filter with cutoff {} unstable with '
                         'sampling frequency {}'.format(cutoff, sampling_freq))
        
    filtered = filtfilt(b, a, signal, method='gust')
    
    return filtered

# =============================================================================
# %% functions for traversing lists
# =============================================================================

def which_one_in(l, f):
    """
    returns which one element of list l is in f, otherwise None
    """

    included = [i for i in l if str(i) in f]
    
    if len(included) == 1:
        return included[0]
    elif len(included) > 1:
        return False
    else:
        return None
    
# =============================================================================
# %% simple exponential decay with x0 = 0, and baseline
# =============================================================================
    
def exp_decay(x, c0, tau, c1=0):
    return c0 * np.exp(-x/tau) + c1

def exp_decay_cumsum(x, c0, tau, c1=0, c2=0):
    return np.cumsum(c0 * np.exp(-x/tau) + c1)+c2

# =============================================================================
# %% double exponential decay with x0 = 0, and baseline
# =============================================================================

def two_exp_decay(x, c0, c1, tau0, tau1, y0=0):
    return y0 + exp_decay(x, c0, tau0) + exp_decay(x, c1, tau1)

def two_exp_decay_cumsum(x, c0, c1, tau0, tau1, y0=0, y1=0):
    return np.cumsum(y0 + exp_decay(x, c0, tau0) + exp_decay(x, c1, tau1)) + y1

# =============================================================================
# %% KWW stretched exponential decay with x0 = 0, and baseline
# =============================================================================
    
def kww_decay(x, c0, tau, beta=1., c1=0):
    return c0 * np.exp(-(x/float(tau))**float(beta)) + c1

# =============================================================================
# %% an erf rise wrapper for fitting functions
# =============================================================================
    
def rising_exp_decay(x, c0, tau, c1=0, x0=0, rt=np.inf):
    return exp_decay(x,c0,tau,c1) * erf(rt*(x-x0))
    
def rising_exp_decay_cumsum(x, c0, tau, c1=0, c2=0, x0=0, rt=np.inf):
    return np.cumsum(exp_decay(x,c0,tau,c1) * erf(rt*(x-x0))) +c2

def rising_two_exp_decay(x, c0, c1, tau0, tau1, y0=0, x0=0, rt=np.inf):
    return two_exp_decay(x, c0, c1, tau0, tau1, y0) * erf(rt*(x-x0))

def rising_two_exp_decay_cumsum(x, c0, c1, tau0, tau1, y0=0, y1=0, x0=0, rt=np.inf):
    return np.cumsum(two_exp_decay(x, c0, c1, tau0, tau1, y0) * erf(rt*(x-x0))) + y1


# =============================================================================
# %% KWW stretched exponential decay with x0 = 0, and baseline, plus:
# ## tail stretch turns on at x=tstar, returns simple exponential for x<tstar
# =============================================================================
    
def kww_decay_break(x, c0, tau, tstar=0, beta=1., c1=0):
    simple = exp_decay(x[x<tstar],c0,tau,c1)
    c02 = exp_decay(tstar, c0, tau, 0)/kww_decay(tstar, 1., tau, beta, 0)
    stretched = kww_decay(x[x>=tstar], c02, tau, beta, c1)
    
    # print x[x<tstar], simple
    # print x[x>=tstar]-tstar, stretched
    # return np.array( [(simple[i], stretched[i])[x[i]>tstar] for i in range(len(x))] )
    return np.concatenate((simple, stretched), axis=None)

def kww_decay_cumsum(x, c0, tau, tstar=0, beta=1., c1=0, c2=0):
    simple = exp_decay(x[x<tstar],c0,tau,c1)
    c02 = exp_decay(tstar, c0, tau, 0)/kww_decay(tstar, 1., tau, beta, 0)
    stretched = kww_decay(x[x>=tstar], c02, tau, beta, c1)
    
    # print x[x<tstar], simple
    # print x[x>=tstar]-tstar, stretched
    # return np.array( [(simple[i], stretched[i])[x[i]>tstar] for i in range(len(x))] )
    return np.cumsum(np.concatenate((simple, stretched), axis=None))+c2

# =============================================================================
# %% Mittag-Leffler function and a wrapper to fit to a lifetime with it
# =============================================================================

def mittag_leffler(x,a,b,terms):
    '''
    Computes the Mittag-Leffler function: 
    E_a,b(x) = sum_0^terms x**term / gamma(a*term+b)
    This is typically called with x = -(t/tau)**a, hence the wrapper below.
    
    Convergence is limited to when the function is not too small and the value
    of the argument x is not too large. 100-200 terms typically gives precision
    that is good enough for any practical application including fitting. This 
    translates to a reliable range of values for the ML function from around
    1 (argument near zero) to 0.01 at best. If you think you need to compute 
    the ML function when it is small (e.g. 1e-3), most likely an approximation
    will work just as fine. 
    
    Sokolov & Klafter in "First Steps in Random Walks" set b=1.
    The Havriliak-Negami relaxation's FT is close to the ML function, but not 
    exact; Cole-Cole relaxation has b=1 - but still other terms in front too.

    Parameters
    ----------
    x : array or number
        The argument of the function.
    a : numerical
        Typical first parameter of the Mittag-Leffler function.
    b : numerical
        Typical first parameter of the Mittag-Leffler function.
    terms : int
        The number of terms to compute for the series. Exact is infinity. 
        100-200 is typically sufficient, using a number higher than that may 
        hamper convergence.

    Returns
    -------
    output
        Same dimensionality as x.

    '''
    return np.sum(np.array([x**k/gamma(k*a+b) for k in range(terms)]), axis=0)

def mittag_leffler_wrapper(x, tau, a, b, terms):
    return mittag_leffler(-(x/tau)**a, a, b, terms)

# =============================================================================
# %% make freud Voronoi & NearestNeighbors objects from a grid of points (x,y)
# ## this method assumes that the box is square
# ## This is copied from crystal_utils on 2020/07/23
# =============================================================================

def make_voronoi(grid_path, L=1., z=0, debug=False):
    '''input: grid_path is the (relative) path to the file with grid points'''

    pts = pd.read_csv(grid_path)
    gridpts = np.array([pts.x, pts.y]).T
    print(f'read {len(gridpts)} grid points')

    ## help_vor is the inverse of unit cells.
    ## Vertices are maxima of mobile ion probability density
    help_vor = Voronoi(gridpts)

    site_pts = []
    for i in range(len(help_vor.vertices[:, 0])):
        if in_bounds(help_vor.vertices[i, :], np.array([0, 0]), np.array([L, L])):
            site_pts.append((help_vor.vertices[i, 0] - 0.5 * L, help_vor.vertices[i, 1] - 0.5 * L, 0))

    ## remove duplicates around the edges of the box if needed, 
    ## this is using brute force
    to_remove = [];
    thresh = L * 1e-4
    new_site_pts = [];
    for i, pt1 in enumerate(site_pts):
        if i in to_remove: continue
        for j, pt2 in enumerate(site_pts[i + 1:]):
            if L - abs(pt1[0] - pt2[0]) < thresh and abs(pt1[1] - pt2[1]) < thresh:
                # print pt1, pt2, i, j+i+1
                to_remove.append(j + i + 1)
            elif L - abs(pt1[1] - pt2[1]) < thresh and abs(pt1[0] - pt2[0]) < thresh:
                # print pt1, pt2, i, j+i+1
                to_remove.append(j + i + 1)
        new_site_pts.append(pt1)

    print(f'{len(site_pts)} points in bounds, removing {len(to_remove)} of them')

    site_pts = np.asarray(new_site_pts)

    box = freud.box.Box.square(L)
    site_pts = box.wrap(site_pts)
    site_vor = freud.locality.Voronoi(box, 0.5 * L)
    site_vor.compute(system=(box, site_pts))
    numpolys = len(site_vor.polytopes)

    if debug: draw_voronoi(box, site_pts, site_vor.polytopes, draw_points=True,
                           cell_numbers=range(len(site_vor.polytopes)))

    ## points at centers of sites, approx. corresponding to oxygen locations
    help_verts = site_vor.polytopes
    these_sites = []
    for s in help_verts:
        for sh in s:
            these_sites.append([sh[0], sh[1], 0])  ## append avg z given that z was previously lost

    help_verts = np.asarray(these_sites)

    ## initialize nearest-neighbor object
    # nn = freud.locality.NearestNeighbors(0.2, 4, 1.05)

    return box, site_vor, site_pts, numpolys, help_verts

# =============================================================================
# %% TODO: plot multiple Voronoi lattices from a multi-plane file with hops
# =============================================================================

def draw_lattices():
    
    return False

# =============================================================================
# %% count and output re-fill times for a single or composite plane
# =============================================================================

def count_fill_times(plane_data, numpolys):
    
    ## sort by time for calculating fill times below
    plane_data.sort_values(by='time',axis=0,inplace=True,ascending=True)
    plane_data.reset_index(inplace=True)
    plane_data.drop(['index'],axis=1,inplace=True)
    
    #numsites = len(total_hops.new_cell.unique()) ## use numpolys instead
    fill_sites = np.zeros((numpolys,2)) ## keep track of which ion was here last
    fill_times = list() ## keep track of which site gets which time to tell BR/a-BR apart

    for i, r in plane_data.iterrows():
        
        ## count the non-hop with its (long) residence time
        if r.old_cell == r.new_cell : 
            fill_times.append(np.array([r.new_resid_time,int(r.old_cell),np.nan]))
            continue
    
        ## restart old site's counter. Even if not onwards hop, new ion could fill
        ## the site before the original returns
        fill_sites[int(r.old_cell),:] = np.array([r.time, int(r.ion)])
    
        ## append new site's fill time with now minus last hop out
        if np.array_equal(fill_sites[int(r.new_cell),:], np.array([0,0])) : ## the first hop into new site
            ## count the fill time of the initial site starting at zero
            fill_times.append(np.array([r.time,int(r.old_cell),np.nan]))
        else :
            ## flag back-hops / re-fills immediately
            ## 3rd column is True for re-fills
            fill_times.append([r.time-fill_sites[int(r.new_cell),0],int(r.new_cell),r.ion==fill_sites[int(r.new_cell),1]])
    
        ## more conditions to go here
        ## TODO: add the fill time from last hop into new site to end of simulation
        if (r.rev_hop != 1.) & (r.rev_hop != 0.):
            fill_times.append([r.new_resid_time,int(r.new_cell),np.nan])
    
        if not i % int(len(plane_data)/20) : print(f'{100*i/len(plane_data):.0f}% done')

    fill_times = pd.DataFrame(data=fill_times, columns=['time','site','refill'])

    fill_times.time = np.round(fill_times.time,3)
    fill_times.site = fill_times.site.astype(int)
    
    return fill_times

# =============================================================================
# %% claculate occupancy of high-energy site based on T and fixed stoichiometry
# ## assuming a two-state system with each level having Nsite sites
# ## assuming Nion total ions, of which Nexc are excited to the higher level
# =============================================================================

def two_state_occup(Nion, Nsite, energy, T):
    ''' energy [eV], T [K] ''' 
    
    lb=max(0,Nion-Nsite)
    ub=Nion/2.
    
    f = lambda Nexc : (Nsite-Nexc)*(Nion-Nexc) - np.exp(energy/kb/T)*(Nsite-(Nion-Nexc))*Nexc
    sol = root_scalar(f, bracket=[lb,ub], method='brentq')
    
    return sol.root

# =============================================================================
# %% calculate energy from occupancy 
# ## assuming Nion total ions, of which Nexc are excited to the higher level
# ## and 2 levels with Nsite sites each; energy is in eV based on kb
# =============================================================================

def two_state_energy(Nion, Nsite, Nexc, T, lb=0., ub = 1.):
    ''' bounds [eV], T [K] ''' 
    
    assert (Nexc <= Nion / 2.) or (lb < 0.), 'N excited > 50%, T > inf @ energy > 0'
    assert Nexc > 0., 'N excited should probably be > 0, or else T < 0'
    
    f = lambda energy : (Nsite-Nexc)*(Nion-Nexc) - np.exp(energy/kb/T)*(Nsite-(Nion-Nexc))*Nexc
    sol = root_scalar(f, bracket=[lb,ub], method='brentq')
    
    return sol.root

# =============================================================================
# %% calculate occupancy from energy for a 2-state model w/ distinct #s o/sites
# ## (i.e. degeneracies of the levels 'g' for ground and 'e' for excited)
# ## assuming levels with degeneracy Ns_g and Ns_e, and N_i total ions
# =============================================================================

def two_state_nondeg_occupancy(Ns_g, Ns_e, Nion, energy, T):
    ''' energy [eV], T [K] ''' 
    
    assert Nion < Ns_g+Ns_e, 'too many ions for {} total sites: {}'.format(Ns_g+Ns_e, Nion)
    
    lb=max(0,Nion-Ns_g)         ## minimum number of ions in excited level
    ub=float(Ns_e)/(Ns_e+Ns_g)  ## toward inf T, all levels have same occupancy
     
    f = lambda Nexc : (Nion-Nexc)*(Ns_e - Nexc) - np.exp(energy/kb/T)*Nexc*(Ns_g-(Nion-Nexc))
    sol = root_scalar(f, bracket=[lb,ub], method='brentq')
    
    return sol.root

# =============================================================================
# %% calculate energy from occupancy for a 2-state model w/ distinct #s of sites
# ## (i.e. degeneracies of the levels 'g' for ground and 'e' for excited)
# ## assuming levels with degeneracy Ns_g and Ns_e, and N_i total ions
# =============================================================================

def two_state_nondeg_energy(Ns_g, Ns_e, Ni_g, Ni_e, T, lb=0., ub = 5.):
    ''' bounds [eV], T [K] '''
    
    assert 0 < Ni_g < Ns_g, 'weird ground state: {:.2f} in {}'.format(Ni_g, Ns_g) ## strict < 
    assert 0 < Ni_e < Ns_e, 'weird excited state: {:.2f} in {}'.format(Ni_e, Ns_e) ## strict
     
    f_g = float(Ni_g) / float(Ns_g) ## fraction of filled  ground-state sites
    f_e = float(Ni_e) / float(Ns_e) ## fraction of filled excited-state sites
    
#    f = lambda energy : f_g*(1.-f_e)*Ns_g*Ns_e - np.exp(energy/kb/T)*f_e*(1.-f_g)*Ns_g*Ns_e
    f = lambda energy : f_g*(1.-f_e) - np.exp(energy/kb/T)*f_e*(1.-f_g)
    sol = root_scalar(f, bracket=[lb,ub], method='brentq')
    
    return sol.root
    
# =============================================================================
# %% multi-method wrapper for calculating expectation values over distributions
# ## assumes that the 'dist' passed on is already a Series, not DataFrame
# =============================================================================

def expectation_multi_method(dist, method, aggregated=False, **kwargs):
    
    if method == 'mean':
        return dist.mean(), dist.std()
    else:
        if not aggregated :
            ## make pdf & cdf
            freqs = dist.groupby(dist).agg('count').pipe(pd.DataFrame).rename(columns = {dist.name: 'frequency'}) 
            freqs['pdf'] = freqs['frequency'] / sum(freqs.frequency)
            freqs['cdf'] = freqs.pdf.cumsum()
            
            # print(freqs.head())
        
            ## create PDF from a distribution
            times = freqs.loc[(freqs.index.values > 1.) & (freqs.index.values < 500)].index.values
            pdf = freqs.loc[(freqs.index.values > 1.) & (freqs.index.values < 500)].pdf.values
            pdf_ub = freqs.pdf.min()
        else: 
            times = dist.index.values
            pdf = dist.values.reshape(-1)
            pdf_ub = pdf.min()/100 if pdf.min() > 0 else 1e-8
            
        ## fit simple exponential time to PDF. Format: [pre-exponent, tau, constant offset]
        ub = [1000., 1e5, pdf_ub] ## hard limit: 1e-7
        lb = [1e-4, 1e-3,      0]
        p0 = [1e-2,3,1e-15]
            
        try: 
            popt, pcov = cf(exp_decay,times,pdf, p0=p0, bounds = (lb, ub))
            perr = np.sqrt(np.diag(pcov))
        except ValueError: 
            print('fitting one exponential did not work, trying a faster decay')
            # popt, pcov = cf(exp_decay,times,pdf, p0=[1e-2,0.5,1e-10], bounds = (lb, ub))
            # perr = np.sqrt(np.diag(pcov))
            popt = p0
            
        if method == 'simple' : 
            if 'verbose' in kwargs.keys() and kwargs['verbose'] : return popt, perr
            else : return popt[1], perr[1]
        else:
            
            ## fit stretch tail with a break, p0 = [c0, tau, tstar, beta=1, c1=0]
            ub = [1000., popt[1]*100., 2000., 1,    pdf_ub] ## hard limit: 1e-7
            lb = [1e-4,  popt[1]*0.1, 0.1,  0,    0]
            p0=[1e-2,popt[1],5,0.9,1e-15]
            
            # print('lb:', lb)
            # print('p0:', p0)
            # print('ub:', ub)
            
            popt, pcov = cf(kww_decay_break,times,pdf, 
                            p0=p0, bounds = (lb, ub),
                            max_nfev=1e4)
            perr = np.sqrt(np.diag(pcov))
            if 'verbose' in kwargs.keys() and kwargs['verbose'] : return popt, perr
            else : return popt[1], perr[1]

# =============================================================================
# $$ functions to query z
# =============================================================================


# =============================================================================
# %% correlation factor
# =============================================================================

def avg_cos_hop(rev_hops):
    ''' honeycomb lattice <cos theta> '''
    return (-1. * len(rev_hops[rev_hops==True]) + 0.5 * len(rev_hops[rev_hops==False]))/len(rev_hops)

def corr_factor(rev_hops):
    cos_theta = avg_cos_hop(rev_hops)
    return (1.+cos_theta)/(1.-cos_theta)

# =============================================================================
# %% parse LAMMPS output with multiple RDFs
# ## file structure: 2 lines of comments, then each step with number-of-rows, 
# ## then that many rows: center of bin, then rdf, then coordination
# ## TODO: merge this with the standalone parse_rdf file
# =============================================================================

def parse_rdf(filepath):
    
    bins = list()
    steps = list()
    first_rdf = list()
    
    with open(filepath, 'r') as fp:
        
        ## read the first three sets of comments
        line = fp.readline() #; print line[:-1]
        line = fp.readline() #; print line[:-1]
        line = fp.readline() #; print line[:-1]
        
        ## read first line with first time step and number-of-rows
        ## take off the endline character and split
        line = fp.readline()
        arr = line[:-1].split(' ')
        steps.append(arr[0])
        numrows = s2n(arr[1])
        
        ## get first set of bins
        for i in range(numrows):
            line= fp.readline()
            arr = line[:-1].split(' ')
            
            bins.append(s2n(arr[1]))
            first_rdf.append(s2n(arr[2]))
            ## skip cdf / coordination
            
        ## check
#        print len(bins), len(first_rdf)
        
        ## make a pandas dataframe
        dfdict = {'bins':np.array(bins), '{}'.format(steps[-1]):first_rdf}
        df = pd.DataFrame(data = dfdict)
        
        ## read next time step
        line = fp.readline()
        
        ## loop through all other sets
        while(line) :
            ## parse line with new time step
            arr = line[:-1].split(' ')
            steps.append(arr[0])
            numrows = s2n(arr[1])
            
            rdf = list()
            bins = list()
            
            for i in range(numrows):
                line= fp.readline()
                arr = line[:-1].split(' ')
                
                bins.append(s2n(arr[1]))
                rdf.append(s2n(arr[2]))
                ## skip cdf / coordination
                
            df['{}'.format(steps[-1])]  = np.array(rdf)
            
            ## check
#            if int(steps[-1]) % 1000 == 0:
#                print 'done {} ps'.format(int(steps[-1])/1000)
                
            ## read next time step
            line = fp.readline()
            
    return df.set_index('bins')
            
# =============================================================================
# %% parse non-gaussian parameter output of LAMMPS
# ## TODO: update this as it is deprecated.
# =============================================================================

def parse_a2(fpath):
    
    times = list(); r2s = list(); r4s = list(); a2s = list()
    
#    df = pd.DataFrame(columns=['r2','r4','a2'])
    with open(fpath,'r') as a2in:
        for i in range(3) : a2in.readline() ## skip header
        
        stepline = a2in.readline()
        
        while stepline:
            times.append(int(stepline[:-1].split(' ')[0])/1000.)
            r2s.append(float(a2in.readline()[:-1].split(' ')[1]))
            r4s.append(float(a2in.readline()[:-1].split(' ')[1]))
            a2s.append(float(a2in.readline()[:-1].split(' ')[1]))
            
#            df.iloc[int(step)] = {'r2':r2, 'r4':r4, 'a2':a2}
            stepline = a2in.readline()
            
        return pd.DataFrame({'time':times, 'r2':r2s, 'r4':r4s, 'a2':a2s}).set_index('time')

# =============================================================================
# %% calculate occupancies of sites properly
# ## counting method: total old time + final time
# =============================================================================
    
def site_occupancies(data2, sites, BR_sites, total_time=10000., plane_label=None, 
                     r='some r',verbose=False):
    ''' 
    data2: pandas df with columns: time, old_resid_time, new_resid_time, old_is_BR 
    sites: (sub)set of sites for which to calculate occupancies
    BR_sites: which sites are BR 
    verbose: boolean flag for printing debug statements
    plane_label: has fields m, T1 if given; 
    r: optional for debugging'''
    
    if plane_label is None : m='M'; T1 = 0; ph = 'beta'
    else: m = plane_label.metal; T1 = plane_label.T1; ph = plane_label.phase
    
    ## columns with distances : new and old
    new_r_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    old_r_col = 'old_ox_r' if ph == 'beta' else 'old_mg_count'
            
    ## downselect plane & sites
    data = data2.query(f'{new_r_col} == @r') ## ignores empties, catches non-hops
    data_new = data2.query(f'{new_r_col} == @r & new_cell != old_cell') ## ignores empties, catches non-hops
    data_old = data2.query(f'{old_r_col} == @r & old_cell != new_cell') ## ignores empties
    data_non = data2.query(f'{old_r_col} == @r & {new_r_col} == @r & old_cell == new_cell')
                
    ## count sites
    old_sites = set(data.old_cell.unique())
    new_sites = set(data.new_cell.unique())
#    ions = len(data.query('new_cell != old_cell').ion.unique()) ## cosmetic only
    
    ## subdivide data manually
    data_new_aBR = data_new[~data_new.new_cell.isin(BR_sites)] ## query overcounts non-hops
    data_new_BR  = data_new[ data_new.new_cell.isin(BR_sites)]
    data_old_BR  = data_old[ data_old.old_cell.isin(BR_sites)]
    data_old_aBR = data_old[~data_old.old_cell.isin(BR_sites)]
    
    ## count empty sites: the series for their zero occupancy will be created later
    empties = sites - old_sites - new_sites
    empties_BR = empties.intersection(BR_sites)
    empties_aBR = empties - empties_BR
    
    if verbose: print(f'\n{m} {T1}K r={r}: {len(data_non)} non-hops, {len(empties)} empties')
    
    ## non-hops in time - modified 2020/08/01
    # time_non_BR =  data_non.query('old_is_BR == True ').groupby('new_cell').new_resid_time.sum()
    # time_non_aBR = data_non.query('old_is_BR == False').groupby('new_cell').new_resid_time.sum()
    time_non_BR =  data_non[ data_non.old_cell.isin(BR_sites)].groupby('new_cell').new_resid_time.sum()
    time_non_aBR = data_non[~data_non.old_cell.isin(BR_sites)].groupby('new_cell').new_resid_time.sum()
    
    old_BR_time  =  data_old_BR.groupby('old_cell').old_resid_time.sum()
    old_aBR_time = data_old_aBR.groupby('old_cell').old_resid_time.sum()
                    
    ## adjust for the final time at a final site. Only one sum() b/c each should be only one hop
    final_times_BR  =  data_new_BR.query('rev_hop != True & rev_hop != False ').groupby('new_cell').new_resid_time.sum()
    final_times_aBR = data_new_aBR.query('rev_hop != True & rev_hop != False ').groupby('new_cell').new_resid_time.sum()
                    
    ## add site-specific values using pandas combine, check lengths.
    csum = lambda s1, s2 : s1 + s2 ## need a function that takes two series to pass to df.combine()
    total_BR  = old_BR_time.combine(final_times_BR, csum, fill_value=0)
    total_aBR = old_aBR_time.combine(final_times_aBR,csum, fill_value=0)
    
    ## add non-hops with pandas append
    if not time_non_BR.empty : total_BR  = total_BR.append(time_non_BR)
    if not time_non_aBR.empty: total_aBR = total_aBR.append(time_non_aBR)
    
    ## create series of zeros for empties and append to the main
    if empties_BR:  total_BR  = total_BR.append(pd.Series(data=0,index=empties_BR))
    if empties_aBR: total_aBR = total_aBR.append(pd.Series(data=0, index=empties_aBR))
    
    ## check lengths and bounds
    if verbose: 
        print(' BR: {} sites, max={:.2f}, min={:.2f}'.format(len(total_BR),  total_BR.max(),  total_BR.min()))
        print('aBR: {} sites, max={:.2f}, min={:.2f}'.format(len(total_aBR), total_aBR.max(), total_aBR.min()))
    
    ## add the radius to make it look like it was just done with groupby calls
    total_BR = pd.DataFrame({'total':total_BR/total_time, new_r_col:r, 'site':'BR'})
    total_aBR = pd.DataFrame({'total':total_aBR/total_time, new_r_col:r, 'site':'aBR'})
    
    return total_BR, total_aBR

# =============================================================================
# %% calculate multi-state (10-state) energies from occupancies
# =============================================================================
    
def multi_state_energies(gro, T1, lb=-2., ub=2.):
    ''' gro: list of triples (degeneracy g, distance r, occupancy o) in order
        T1: temperature [K] '''
    gro = sorted(gro, key = lambda x: abs(x[2]-0.5)) ## put closest-to-half-full 1st
    es  = [[] for x in gro]
    es[0].append(0.) ## make a convention
    rs = []
    ab = []
    
    for i, s1 in enumerate(gro):
        rs.append(s1[1])
        try: ab.append(s1[3])
        except: pass
    
        for j, s2 in enumerate(gro): ## no need to double up; use 1 as ground & 2 as exc
            if j <= i : continue
            E21 = two_state_nondeg_energy(s1[0], s2[0], s1[0]*s1[2], s2[0]*s2[2], T1, lb, ub)
            es[j].append(E21+es[i][-1])
    #        print i, j, s1[1], s2[1], np.round(E21,3)
            
    es = [np.mean(x) for x in es]
    
    return pd.DataFrame( {'rs':rs, 'es':es, 'site':ab})

# =============================================================================
# %% calculate multi-state (10-state) energies from occupancies
# ## while keeping site labels intact and attached
# =============================================================================
    
def multi_site_es(gro, T1, lb=-2., ub=2.):
    ''' gro: pandas df with columns: degeneracy g, distance r, occupancy o, 
             site kind 'site', and indexed by site number.
        T1: temperature [K] '''
    gro['o2'] = np.abs(gro.o - 0.5)
    gro.sort_values(by='o2',inplace=True) ## put closest-to-half-full 1st
    gro.drop(columns='o2',inplace=True)
    gro.reset_index(inplace=True)
    gro.rename(columns={'index':'cell'},inplace=True)
    
    es  = dict([(gro.iloc[x].cell,[]) for x in range(len(gro))])
    es[gro.iloc[0].cell].append(0.) ## make a convention / starting point

    try:
        for row1 in gro.itertuples():
            for row2 in gro.itertuples(): ## no need to double up; use row1 as ground & row2 as excited
                if row2.Index <= row1.Index : continue
                E21 = two_state_nondeg_energy(row1.g, row2.g, row1.o*row1.g, row2.g*row2.o, T1, lb, ub)
                es[row2.cell].append(E21+es[row1.cell][-1])
    except:
        print(row2.cell)
        
    es = dict([(k,np.mean(v)) for (k,v) in list(es.items())])
    gro['e'] = gro.cell.map(es)
    
    return gro

# =============================================================================
# %% load planes using the 2020/07/26 framework: return a big dataframe
# ## this requires the input plane to have been a row in big all_planes database
# =============================================================================

def load_plane_with_atoms(plane, frac=False, do_fill_times=False):
    
    print(f'\nplane: {plane.hop_path}')
    
    ## extract all the metadata
    mm = plane.metal
    TK = int(plane.T1)
    zz = plane.z
    st = plane.stoich
    ex = plane.exclude
    ph = plane.phase
    tt = plane.total_time
    cn = plane.config ## config is st + ex together
    
    ## TODO: load everything that is auxiliary using the lammps file:
    ## mobile ion sites: site_pts, locations of defects, BR sites

    ## Load the lammps file
    _, _, cell, all_atoms = cu.read_lmp(plane.lammps_path, fractional=frac)
    conduction_planes = cu.get_conduction_planes(all_atoms, mm, inexact=False if ph == 'beta' else True)
    planes_by_name = dict(zip([cu.standard_plane_name(z if frac else z/cell[2,2]) for z in conduction_planes], conduction_planes))
    
    ## separate oxygen defects: remove the max() of atom types that are oxygen
    ## and store the oxygens separately
    if ph == 'beta' :
        if len(all_atoms.query('atom == "O"').type.unique() ) > 1:
            type_ointer = all_atoms.query('atom == "O"').type.max()
            ointers = all_atoms.query(f'type == {type_ointer}')
            atoms = all_atoms.query(f'type != {type_ointer}')
            print(f'found {len(ointers)} defects')
        else : 
            ointers = pd.DataFrame()
            atoms = all_atoms
    else :
        ## doubleprime: load distances to Mg instead
        ointers = pd.DataFrame()
        atoms = all_atoms
        
    ## Mg defects should be checked for both beta and beta"
    mgs = atoms.query('atom == "Mg"')
    mgs = atoms.query('type == 4 & atom != "O"')
        
    ## count the planes
    ## count the mobile-ion sites 
    num_planes = len(conduction_planes) if zz == 'z_all' else int(plane.num_planes)
    dz = np.mean(np.diff(conduction_planes))
    
    ## Initialize variables for advanced data
    all_sites_by_r = list()
    edge_matrices = list()
    all_BR_sites = list()
    defect_coords = list()
    
    ## compute 2D matrix of from-to distances to defects
    if num_planes == 1 and ph == 'beta':
        all_site_pts = cu.get_mobile_ion_sites(atoms, planes_by_name[zz], cell if not frac else np.eye(3))
        _, edges, _ = cu.get_mid_oxygen_sites_freud(all_site_pts, cell, viz=False)
        
        num_polys = len(all_site_pts)
        
        ## auto-get BR sites - this just yields a list of false for beta-doubleprime
        site_types = cu.auto_get_BR_sites(atoms, cell, all_site_pts, atoms_are_frac=frac)
        all_BR_sites = [i for i in range(len(site_types)) if site_types[i] == 'BR']
        print(f'{len(all_BR_sites)} BR sites')
        
        ## make graph path lengths
        nxg = nx.from_edgelist(edges)
        paths = cu.path_lengths(nxg)
        
        ## calculate distances to defects
        if len(ointers) > 0 : 
            ## get all sites and edges
            oi_defects = ointers.query(f'z == {planes_by_name[zz]}')
            oi_defect_pts = oi_defects[['x','y','z']].values
            oi_adjacent_sites, _ = cu.get_defect_adjacent_sites(cell, all_site_pts, oi_defect_pts)
            defect_coords = oi_defects
            # print(f'sites next to O_i in plane {zz}: ', oi_adjacent_sites)
            
            # calculate paths to interstitial-adjacent mobile-ion sites
            paths_to_oi = [int(min(paths[oi_adjacent_sites,x])) for x in range(num_polys)]
            max_r = max(paths_to_oi)
            
            ## bin sites by distance
            for r in range(max_r+1):
                all_sites_by_r.append(np.argwhere(np.array(paths_to_oi) == r).T[0])
            
            ## store paths to oxygens as a matrix
            edge_matrix = np.zeros((max_r+1, max_r+1))
            for edge in np.argwhere(paths == 1) :
                edge_matrix[paths_to_oi[edge[0]], paths_to_oi[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
        
        ## if there are Mg defects instead of Oi, then count distances to them
        elif len(mgs) > 0 : 
            ## create a placeholder for the coordinates of the created defects
            defect_pts = mgs[['x', 'y', 'z']].values
            
            ## find the Mg closest to each mobile-ion site in this plane
            e0, e1, d0, d1 = cu.get_nearest_points(all_site_pts, defect_pts, cell, num_nn=6)
            e0 = np.array(e0)[np.array(d0)<dz]
            e1 = np.array(e1)[np.array(d1)<dz]
            
            ## indices of mobile-ion sites
            s0 = [x[1] for x in e0]
            s1 = [x[1] for x in e1]
            
            ## Mg locations
            mg0 = [x[0] for x in e0]
            mg1 = [x[0] for x in e1]
            defect_coords = mgs.iloc[list(set(mg0+mg1))]
            
            # it will be more tidy for combining distances later to keep placeholder arrays
            if len(s0) > 0: paths_to_mg_0 = [min(paths[s0, x]) for x in range(len(all_site_pts))]
            else: paths_to_mg_0 = np.ones(len(all_site_pts))*len(all_site_pts)
            
            if len(s1) > 0: paths_to_mg_1 = [min(paths[s1, x])+1 for x in range(len(all_site_pts))]
            else: paths_to_mg_1 = np.ones(len(all_site_pts))*len(all_site_pts)
            
            # combine path lengths to distance==1 and distance==0 sites using min()
            paths_to_oi = [int(min(paths_to_mg_0[i], paths_to_mg_1[i])) for i in range(len(all_site_pts))]
            max_r = max(paths_to_oi)
            
            ## bin sites by distance
            for r in range(max_r+1):
                all_sites_by_r.append(np.argwhere(np.array(paths_to_oi) == r).T[0])
            
            ## store paths to oxygens as a matrix
            edge_matrix = np.zeros((max_r+1, max_r+1))
            for edge in np.argwhere(paths == 1) :
                edge_matrix[paths_to_oi[edge[0]], paths_to_oi[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
            
        # if there are no oxygens and no Mg, then make very long paths
        else :
            print(f'(whether intentional or not, no O_i in plane {zz})')
            paths_to_oi = np.ones(num_polys).astype(int)*100
            ## TODO: make placeholders for sites_by_r and edge_distances for defect-free
            # max_r = max(paths_to_oi)
        
    ## this needs to be broken down by plane for composite planes
    elif ph == 'beta': 
        num_polys = 0
        max_r = 0
        all_paths_to_oi = list()
        all_site_pts = list()
        paths_list = list()
        
        for c, p in enumerate(conduction_planes) :
            ## get all sites and edges
            site_pts = cu.get_mobile_ion_sites(atoms, p, cell if not frac else np.eye(3))
            _, edges, _ = cu.get_mid_oxygen_sites_freud(site_pts, cell, viz=False)
            num_polys += len(site_pts)
            all_site_pts.append(site_pts)
            
            ## auto-get BR sites - this just yields a list of false for beta-doubleprime
            site_types = cu.auto_get_BR_sites(atoms, cell, site_pts, atoms_are_frac=frac)
            all_BR_sites.append([s+len(site_pts)*c for s in range(len(site_types)) if site_types[s] == 'BR'])
            
            ## make graph path lengths
            nxg = nx.from_edgelist(edges)
            paths = cu.path_lengths(nxg)
            paths_list.append(paths)
            # print(f'found paths for plane {p:.4f}')
            
            ## calculate distances to defects
            if len(ointers) > 0 : 
                ## get all sites and edges
                oi_defects = ointers.query(f'z == {p}')
                oi_defect_pts = oi_defects[['x','y','z']].values
                oi_adjacent_sites, _ = cu.get_defect_adjacent_sites(cell, site_pts, oi_defect_pts)
                defect_coords.append(oi_defects)
                # print(f'sites next to O_i in plane {p:.4f}: ', oi_adjacent_sites)
                
                # calculate paths to interstitial-adjacent mobile-ion sites
                paths_to_oi = [int(min(paths[oi_adjacent_sites,x])) for x in range(len(site_pts))]
                all_paths_to_oi.append(paths_to_oi)
                
                max_r = max(max_r, max(paths_to_oi))
                # print(f'farthest distance: {max_r}')
                
                ## store paths to oxygens as a matrix
                edge_matrix = np.zeros((max_r+1, max_r+1))
                for edge in np.argwhere(paths == 1) :
                    edge_matrix[paths_to_oi[edge[0]], paths_to_oi[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
                edge_matrices.append(edge_matrix)
                # print('added a matrix of edges')
                
            ## if there are Mg defects instead of Oi, then count distances to them
            elif len(mgs) > 0 :
                ## create a placeholder for the coordinates of the created defects
                defect_pts = mgs[['x', 'y', 'z']].values
                
                ## find the Mg closest to each mobile-ion site in this plane
                e0, e1, d0, d1 = cu.get_nearest_points(site_pts, defect_pts, cell, num_nn=6)
                e0 = np.array(e0)[np.array(d0)<dz]
                e1 = np.array(e1)[np.array(d1)<dz]
                
                ## indices of mobile-ion sites
                s0 = [x[1] for x in e0]
                s1 = [x[1] for x in e1]
                
                ## Mg locations
                mg0 = [x[0] for x in e0]
                mg1 = [x[0] for x in e1]
                defect_coords.append(mgs.iloc[list(set(mg0+mg1))])
                
                # it will be more tidy for combining distances later to keep placeholder arrays
                if len(s0) > 0: paths_to_mg_0 = [min(paths[s0, x]) for x in range(len(site_pts))]
                else: paths_to_mg_0 = np.ones(len(site_pts))*len(site_pts)
                
                if len(s1) > 0: paths_to_mg_1 = [min(paths[s1, x])+1 for x in range(len(site_pts))]
                else: paths_to_mg_1 = np.ones(len(site_pts))*len(site_pts)
                
                # combine path lengths to distance==1 and distance==0 sites using min()
                paths_to_oi = [int(min(paths_to_mg_0[i], paths_to_mg_1[i])) for i in range(len(site_pts))]
                all_paths_to_oi.append(paths_to_oi)
                
                max_r = max(max_r, max(paths_to_oi))
                # print(f'farthest distance: {max_r}')
                
                ## store paths to oxygens as a matrix
                edge_matrix = np.zeros((max_r+1, max_r+1))
                for edge in np.argwhere(paths == 1) :
                    edge_matrix[paths_to_oi[edge[0]], paths_to_oi[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
                edge_matrices.append(edge_matrix)
                
            # if there are no oxygens and no Mg, then make very long paths
            else:
                print(f'(whether intentional or not, no O_i in plane {p:.4f})')
                paths_to_oi = np.ones(len(site_pts)).astype(int)*100
        
        ## add all edge matrices together
        edge_matrix = np.zeros((max_r+1, max_r+1))
        for mat in edge_matrices: edge_matrix[:len(mat),:len(mat)] += mat
        
        ## bin sites by distance, add to list
        all_paths_to_oi = list(cu.flatten(all_paths_to_oi))
        for r in range(max_r+1):
            all_sites_by_r.append(np.argwhere(np.array(all_paths_to_oi) == r).T[0])
            
        ## flatten the nested list of BR sites
        all_BR_sites = list(cu.flatten(all_BR_sites))
        print(f'{len(all_BR_sites)} BR sites')
        
        ## combine all paths into a large paths array
        paths = np.zeros((num_polys, num_polys)) + num_polys
        prev_sites = 0
        for pz, paths_matrix in zip(conduction_planes, paths_list):
            these_sites = len(paths_matrix)
            paths[prev_sites:prev_sites + these_sites, prev_sites:prev_sites + these_sites] = paths_matrix
            prev_sites += these_sites
            print(f'plane at {pz:.4f} : {these_sites} sites with max {prev_sites}')
    ## doubleprime, single plane
    ## fields to use: max_r, edge_matrix, all_BR_sites, all_sites_by_r,
    ##                defect_coords, all_site_pts, paths
    elif num_planes == 1 :
        
        ## get sites & paths
        all_site_pts, _, _, paths = cu.get_sites_above_below(planes_by_name[zz], atoms, \
                cell if not frac else np.eye(3), metal=mm, frac=frac, viz=False)
    
        num_polys = len(all_site_pts)
        
        ## get defect coordinates
        defect_coords = cu.get_nearby_atoms_wrap(planes_by_name[zz], mgs, 
                            np.mean(np.diff(conduction_planes)), cell[2,2])
        mg_pts = defect_coords[['x','y','z']].values
        
        ## paths to defects, and count defects
        paths_to_mg, mg_counts = cu.get_mg_distances(all_site_pts, paths, mg_pts, cell if not frac else np.eye(3))
        max_r = max(mg_counts)
        
        ## bin sites by the number of nearest defects rather than by distance
        for r in range(max_r+1):
            all_sites_by_r.append(np.argwhere(np.array(mg_counts) == r).T[0])
            
        ## store counts of defects for every edge between sites as a matrix
        edge_matrix = np.zeros((max_r+1, max_r+1))
        for edge in np.argwhere(paths == 1) :
            edge_matrix[mg_counts[edge[0]], mg_counts[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
            # pass
        
        ## pretend that 0 and 1 distances are BR and aBR
        all_BR_sites = np.argwhere(np.array(paths_to_mg) == 0).T[0]
        
    ## doubleprime, combined planes. Aggregate all_site_pts from site_pts, 
    ## add all edge matrices together, bin sites by distances & add, combine paths
    else :
        
        num_polys = 0
        max_r = 0
        paths_list = list()
        all_mg_counts = list()
        all_site_pts = list()
        
        # conduction_planes = sorted(conduction_planes, reverse=True)
        
        for c, p in enumerate(conduction_planes):
            # print(f'\n  starting {c} : {p:.2f}')
            
            ## get sites & paths
            site_pts, _, _, paths = cu.get_sites_above_below(p, atoms, \
                    cell if not frac else np.eye(3), metal=mm, frac=frac, viz=False)
            
            num_polys += len(site_pts)
            all_site_pts.append(np.copy(site_pts))
            paths_list.append(np.copy(paths))
            
            # print(f' got paths {c} : {p:.4f}')
            
            ## get defect coordinates
            mg_coords = cu.get_nearby_atoms_wrap(p, mgs, np.mean(np.diff(conduction_planes)), cell[2,2])
            mg_pts = mg_coords[['x','y','z']].values
            defect_coords.append(mg_coords.copy(deep=True))
            
            # print(f'got defects {c} : {p:.4f}')
            
            ## paths to defects, and count defects
            paths_to_mg, mg_counts = cu.get_mg_distances(site_pts, paths, mg_pts, 
                                        cell if not frac else np.eye(3), verbose=False)
            max_r = max(max_r, max(mg_counts))
            all_mg_counts.append(mg_counts)
            
            # print(f'got dists {c} : {p:.4f}')
            
            ## bin sites by the number of nearest defects rather than by distance
            # for r in range(max_r+1):
            #     sites_by_r.append(np.argwhere(np.array(mg_counts) == r).T[0])
                
            ## store counts of defects for every edge between sites as a matrix
            edge_matrix = np.zeros((max_r+1, max_r+1))
            for edge in np.argwhere(paths == 1) :
                edge_matrix[mg_counts[edge[0]], mg_counts[edge[1]]] += 0.5 if edge[0] != edge[1] else 1
            edge_matrices.append(edge_matrix)
            
            ## pretend that 0 and 1 distances are BR and aBR
            BR_sites = np.argwhere(np.array(paths_to_mg) == 0).T[0]
            all_BR_sites.append(BR_sites+len(site_pts)*c)
            
            # print(f' finishing {c} : {p:.4f}')
            
            del site_pts, paths, mg_coords
            
        # print('putting it together')
        ## make sites_by_r
        ## bin sites by distance, add to list
        all_mg_counts = list(cu.flatten(all_mg_counts))
        for r in range(max_r+1):
            all_sites_by_r.append(np.argwhere(np.array(all_mg_counts) == r).T[0])
            print(f'{len(all_sites_by_r[r]):4d} sites w/ {r} Mg_Al neighbors')
        
        ## add all edge matrices together
        edge_matrix = np.zeros((max_r+1, max_r+1))
        for mat in edge_matrices: edge_matrix[:len(mat),:len(mat)] += mat
        
        ## flatten the nested list of BR sites
        all_BR_sites = list(cu.flatten(all_BR_sites))
        print(f'{len(all_BR_sites)} sites w/ Mg_Al directly above/below (accessible as BR_sites)')
        
        ## combine all paths into a large paths array
        paths = np.zeros((num_polys, num_polys)) + num_polys
        prev_sites = 0
        for pz, paths_matrix in zip(conduction_planes, paths_list):
            these_sites = len(paths_matrix)
            paths[prev_sites:prev_sites + these_sites, prev_sites:prev_sites + these_sites] = paths_matrix
            prev_sites += these_sites
            print(f'plane at z = {pz:.4f} : {these_sites} sites with max {prev_sites}')
    
    ## load plane to get statistics
    th, _, nh, _, _ = load_plane(plane.hop_path, numpolys=num_polys)
    data = pd.concat([th,nh],ignore_index=True)
    
    ## TODO: deal with Voronoi objects using the lammps file
    if s2n(zz) : ## one plane: Voronoi implemented
        pass
    elif zz == 'z_all' : 
        pass
    
    ## add the counting of fill times. 
    ## This is separate, because filling time does not correspond 1-1 to hops
    fill_times = None
    if do_fill_times:
        hop_folder = '/'.join(plane.hop_path.split('/')[:-1])
        fill_time_path = hop_folder+f'/{mm}{TK}K_{zz}_fill_times.csv'
        if path.isfile(fill_time_path) and path.getsize(fill_time_path) > 0:
            print('found pre-processed fill times')
        else :
            print('processing fill times...')
            ft = count_fill_times(data,num_polys)
            ft.to_csv(fill_time_path, index=False)
            print(f'... saved fill times to {fill_time_path}')
        
        ## actually load the fill times, add BR/aBR label, add distance to defect
        fill_times = pd.read_csv(fill_time_path).sort_values(by='site')
        fill_times['site_is_BR'] = [x in all_BR_sites for x in fill_times.site]
        try: fill_times['r_to_defect'] = [all_paths_to_oi[x] if ph == 'beta' else all_mg_counts[x] for x in fill_times.site.values ]
        except: fill_times['r_to_defect'] = 100

    ## return a dictionary with all the data and metadata
    return dict(phase=ph, config=cn, stoich=st, exclude=ex, metal=mm, T1=TK, 
                z=zz, hops=data, atoms=atoms, max_r=max_r, total_time=tt,
                edge_distances=edge_matrix, BR_sites=all_BR_sites,
                sites_by_r=all_sites_by_r, defects=defect_coords,
                num_planes=num_planes, cell=cell, atoms_are_frac=frac,
                site_pts=all_site_pts, path_lengths=paths, hop_path=plane.hop_path,
                fill_times=fill_times)

# =============================================================================
# %% a full autocorrelation function: assumes input has columns vx, vy, vz
# ## (so I guess a velocity autocorrelation function, not position)
# ## For a parallelized version, see 
# =============================================================================

def autocorrelation(df, time_limit, dims=['x','y','z'], verbose=False, to_file=None):
    '''
    '''
    
    start_time = dt.now()

    time = df.index.values
    num_periods = np.argwhere(time > time_limit)[0,0]
    df = df[['vx','vy','vz']]
    
    for d in ['x','y','z'] : 
        if d not in dims : df.drop(f'v{d}', axis=1, inplace=True)
    
    acf = list()
    
    if to_file is not None : 
        fout = open(to_file,'w')
        fout.write('time,' + ','.join(sorted(dims))+'\n')
    
    for lag in range(num_periods):
        cfun = dict(zip(sorted(dims),df.apply(lambda col: col.autocorr(lag))))
        acf.append(cfun)
        if verbose and np.isclose(time[lag] % 25, 0, atol=1e-4) :
            print(f'done lag {time[lag]:.3f} ps, {dt.now()-start_time}')
        
        if to_file is not None:
            fout.write(','.join([str(time[lag]), ','.join([str(cfun[x]) for x in sorted(dims)]) ])+'\n')
        
    acf = pd.DataFrame(acf, index=time[:num_periods]) #.set_index('time')
    
    if to_file is not None: fout.close()
    
    return acf
    
# =============================================================================
# %% apply a function to a one-atome trajectory for an arbitrary # starts
# =============================================================================

def multistart_apply(traj, duration, starts,func=lambda x: pow(x,2)):
    '''
    traj: pandas DataFrame with columns x,y,z and index time
    duration: for how long to count 
    func: function to apply element-wise, default x**2 for r2. r4 is the second obvious use
    '''
    num_points = sum(traj.index.values <= starts[0]+duration)
    
    ## build 3D array
    traj_array = np.zeros((len(starts),num_points,3))
    for i, start in enumerate(starts): 
        traj_array[i,:,:] = traj.loc[start:start+duration,['x','y','z']].values
        
    ## apply the function, then sum over last axis (which is xyz)
    ## then average over the first axis (which is over the multiple starts)
    rn = np.mean(np.sum(func(traj_array), axis=-1), axis=0)
    
    ## make a dataframe
    new_index = traj.index.values-traj.index.values[0]
    new_index = new_index[:len(rn)]
    
    return pd.DataFrame(data={'r2':rn, 'time':new_index})

# =============================================================================
# %% calculate r2 for one atom given trajectory for an arbitrary # starts
# =============================================================================

def multistart_r2r4(traj, duration, starts, cell=np.eye(3), timebins=None, do_avg=True, col='r2', twod=False):
    '''
    traj: pandas DataFrame with columns x,y,z and index time
    duration: for how long to count 
    starts: iterable of time values over which to average
    cell: simulation box, assumed orthogonal
    timebins: time values for binning (e.g. logspace)
    do_avg : boolean flag to perform averaging over the multiple starts
    col : option for which column to output, e.g. 'r2' , 'dx', 'dy' etc
    twod : boolean flag, set to True if z is to be ignored
    '''
    # num_points = sum(traj.index.values <= starts[0] + duration)
    
    if timebins is None : 
        num_points = sum(traj.index.values <= starts[0] + duration)
    else :
        duration = max(timebins)
        num_points = len(timebins) # -1
        
        ## repeat precision determination for rounding just below
        traj.index -= traj.index.min()
        ts = traj.index[1] - traj.index[0]
        prec = -int(np.floor(np.log10(ts)))
        while not np.isclose(ts % 10**-prec, 0) : prec += 1
        
        # make a mask for which values to take 
        mask_all = np.in1d(np.round(traj.index.values,prec), timebins) ## -min(traj.index.values)
        mask = mask_all[:len(traj.loc[:duration])]
        
        print(sum(mask), len(mask), len(timebins))
        
        if len(timebins) > sum(mask) :
            print('extra time bins getting skipped:', sorted(list(set(timebins)-set(mask[mask==1]))))
            num_points=sum(mask)
        
    if num_points < 1 : num_points = 1
    
    assert col == 'r2' or col in dims.keys(), 'Passed value for "col" is not recognized. Passed: {col}. Recognized: "r2" or in {dims.keys()}'
    
    ## take differences for later computing r2
    traj['dx'] = traj.x.diff() * cell[0,0]
    traj['dy'] = traj.y.diff() * cell[1,1]
    traj['dz'] = traj.z.diff() * cell[2,2]
    traj.iloc[0,-3:] = 0
    
    ## account for atoms hopping the border of the cell
    traj.dx[traj.dx> 0.5*cell[0,0]] -= cell[0,0] ## left x boundary crossed
    traj.dy[traj.dy> 0.5*cell[1,1]] -= cell[1,1] ## left y boundary crossed
    traj.dz[traj.dz> 0.5*cell[2,2]] -= cell[2,2] ## left z boundary crossed
    traj.dx[traj.dx<-0.5*cell[0,0]] += cell[0,0] ## right x boundary crossed
    traj.dy[traj.dy<-0.5*cell[1,1]] += cell[1,1] ## right y boundary crossed
    traj.dz[traj.dz<-0.5*cell[2,2]] += cell[2,2] ## right z boundary crossed
    
    ## build 3D array of r2 using cumsum
    traj_array = np.zeros((len(starts),num_points, 2 if twod else 3))
    
    # print(traj_array.shape)
    
    for i, start in enumerate(starts):
        if twod: chunk = traj.loc[start-5e-4:start+duration+5e-4,['dx','dy']].values
        else : chunk = traj.loc[start-5e-4:start+duration+5e-4,['dx','dy','dz']].values
        
        chunk[0,:] = 0
        chunk = np.cumsum(chunk, axis=0)
        
        # if i < 1 : print(chunk.shape, mask.shape, traj_array.shape)
            
        traj_array[i,:,:] = chunk if timebins is None else chunk[mask]
        
        if i < 1 : print('at least one of multiple starts works.')
                
        
    ## apply the square, then sum over last axis (which is x,y,z)
    ## this gets the distribution of r2
    r2 = np.sum(traj_array**2, axis=-1)
    # print(f'r2 is {r2.shape}')
    
    ## bin by time if time bins exist
    if timebins is not None :        
        # new_index = np.array(timebins[:-1])*0.5 + np.array(timebins[1:])*0.5
        
        new_index = traj.index.values-traj.index.values[0]
        new_index = new_index[mask_all]
        
    else :
        new_index = traj.index.values-traj.index.values[0]
        new_index = new_index[:len(r2[0,:])]
    
    if do_avg :
        ## average over the first axis (which is over the multiple starts) to get <r2>
        exp_r2 = np.mean(r2, axis=0)
        
        ## calculate r4 from the distribution first, then average
        exp_r4 = np.mean(r2**2, axis=0)
        
        ## make the output dataframe
        out = pd.DataFrame(data={'r2':exp_r2, 'r4':exp_r4, 'time':new_index})
    else : 
        # print(r2.shape, len(timebins), len(starts))
        out = pd.DataFrame(r2.T) if col == 'r2' else pd.DataFrame(traj_array[:,:,dims[col]].T)
        out['time'] = new_index

        out.set_index('time', inplace=True)
        out.columns = starts
    
    return out #, traj.loc[:,['dx','dy','dz']]

# =============================================================================
# %% calculate r2 for one atom given trajectory for an arbitrary # starts
# =============================================================================

def multiduration_r2r4(traj, deltas, starts, cell=np.eye(3), discard=4):
    '''
    traj: pandas DataFrame with columns x,y,z and index time
    deltas: time lags for computing r2 and r4, the longest is taken
    starts: iterable of time values over which to average, typically a range()
    cell: simulation box, assumed orthogonal
    discard: toss out this multiple of short-time averages-over-duration
    '''
    ## 
    try: duration = max(deltas)
    except: duration = int(deltas)
    
    # if not isinstance(starts, list) : 
    starts = np.array(starts)
    # if min(starts) < duration * discard : starts += duration * discard - min(starts)
        
    ## number of time points in one lag-length of the trajectory
    num_points = sum((traj.index.values >= starts[0]) & (traj.index.values <= starts[0] + duration)) # if timebins is not None else len(timebins)-1
    if num_points < 1 : num_points = 1
    
    ## take differences for later computing r2
    traj['dx'] = traj.x.diff() * cell[0,0]
    traj['dy'] = traj.y.diff() * cell[1,1]
    traj['dz'] = traj.z.diff() * cell[2,2]
    traj.iloc[0,-3:] = 0
    
    ## account for atoms hopping the border of the cell
    traj.dx[traj.dx> 0.5*cell[0,0]] -= cell[0,0] ## left x boundary crossed
    traj.dy[traj.dy> 0.5*cell[1,1]] -= cell[1,1] ## left y boundary crossed
    traj.dz[traj.dz> 0.5*cell[2,2]] -= cell[2,2] ## left z boundary crossed
    traj.dx[traj.dx<-0.5*cell[0,0]] += cell[0,0] ## right x boundary crossed
    traj.dy[traj.dy<-0.5*cell[1,1]] += cell[1,1] ## right y boundary crossed
    traj.dz[traj.dz<-0.5*cell[2,2]] += cell[2,2] ## right z boundary crossed
    
    ## build 3D array of r2 using cumsum
    traj_array = np.zeros((len(starts),num_points,3))
    for i, start in enumerate(starts):
        chunk = traj.loc[start:start+duration,['dx','dy','dz']].values
        chunk[0,:] = 0
        traj_array[i,:,:] = np.cumsum(chunk, axis=0) 
        
    ## apply the square, then sum over last axis (which is dx,dy,dz)
    ## this gets the distribution of r2 and is a 2D array with shape (starts, time points)
    ## here, only the last time point is kept
    r2 = np.sum(traj_array[:,-1,:]**2, axis=-1)
    # print(f'r2 is {r2.shape}')
    
    ## starts_to_count is actually t minus delta in eq.1 from He(2008)
    starts_to_count = np.argwhere(starts>=duration*discard)
    this_r2 = (np.cumsum(r2)/ np.arange(1,1+len(r2)) )
    this_r2 = this_r2[starts_to_count].reshape(-1)
    this_r4 = this_r2**2 
    
    out = pd.DataFrame(data={'r2':this_r2, 'r4':this_r4, 'time':duration+starts[starts_to_count].reshape(-1)})
    # out.time = out.time.astype(int)
    
    ## TODO: downselect time points with a mask

    return out

# =============================================================================
# %% calculate the fluctuation kernel for the diffusion coefficient
# ## using Laplace transforms
# =============================================================================

laplace = lambda x, y, s : np.trapz(y * np.exp(-s * x), x)

def fluctuation_kernel(a2, s_values, dim=3):
    '''

    Parameters
    ----------
    a2 : pandas DataFrame
        a2 is an output from MD, should be indexed to time and have fields r4 and r2.
    s_values : numpy array
        Values of the Laplace-space variable to use.
    dim : int, optional
        Number of dimensions. The default is 3.

    Returns
    -------
    cds : TYPE
        DESCRIPTION.

    '''
    if 0 not in a2.index.values : 
        a2.loc[0] = np.zeros(len(a2.columns))
        a2 = a2.sort_index()
    a2['r22'] = a2.r2 **2
    a2['x4'] = a2.r4-(dim+2)*a2.r22/dim
    a2['burn'] = 0
    
    dx4dt = np.diff(a2.x4,1)/np.diff(a2.index.values,1)
    time_midpoints = a2.index.values[1:] - a2.index.values[:1]
    a2.burn.iloc[1:-1] = np.diff(dx4dt,1)/np.diff(time_midpoints,1) / 24

    # dt = a2.index.values[1] - a2.index.values[0]
    # a2.burn.iloc[1:-1] = np.diff(np.diff(a2.x4,1),1)/dt**2 / 24
    
    cd = list()
    for s in s_values:
        cds  = 3*laplace(a2.index, a2.burn, s)/(dim+2)          ## BCF term
        cds += s**2 * laplace(a2.index, a2.r22, s) / 8 / dim**2 ## r22 term
        cds -= s**3 * laplace(a2.index, a2.r2, s)**2 / 4 / dim**2 ## (r2)^2 term
        cds /= (s**2 * laplace(a2.index, a2.r2, s) / 2 / dim)**2  ## Dgs^2 term
        cd.append(cds)
        
    int1s = interp1d(s_values, np.array(cd), bounds_error=False, fill_value='extrapolate')
    # cd_interp = lambda x: int1s(x)
    
    # cd2 = [ invertlaplace(cd_interp, x, method='dehoog', dps=5, degree=5) for x in a2.index.values[1::10].tolist() ]
    
    # return a2.index.values[1::10], cd2
    return int1s
    
# =============================================================================
# %% Stehfest algorithm for the inverse Laplace transform
# ## https://gist.github.com/AndrewWalker/5583653 
# ## another method: https://github.com/mojtaba-komeili/numerical-inverse-laplace/blob/master/NumInvLaplace.py
# =============================================================================

from math import factorial

def stehfest_coeff(n, i):
    acc = 0.
    for k in range(int(np.floor((i+1)/2.0)), int(min(i, n/2.0))+1) :
        num = k**(n/2.0) * factorial(2 * k)
        den = factorial(i - k) * factorial(k -1) * factorial(k) * factorial(2*k - i) * factorial(n/2.0 - k)
        acc += (num /den)
    exponent = i+n/2.0
    term = np.power(-1+0.0j,exponent)
    res = term * acc
    return res.real

def stehfest_inverse(f, t, n=6):
    acc = 0.
    lton2 = np.log(2) / t
    for i in range(1, n+1):
        a = stehfest_coeff(n, i)
        b = f(i * lton2)
        acc += (a * b)
    return lton2 * acc

# =============================================================================
# %% 2-way smoothing for plotting hopping PDF, CDF, and filling times
# ## 2-way is for forward & back smoothing. Could use a convolution instead.
# =============================================================================

def pdf_smooth(df, halflife):
    ## assume series index is in [picoseconds], as it is for hopping residence times.
    ## this only works approx 59 [picoseconds] at a time - so needs recursion
    
    # ## keeping real [ps] units doesn't work as pandas uses [nanosecond] precision ... sigh ...
    # df.index = [np.datetime64(int(x*1000),'fs') for x in df.index.values]
    
    ## get everything int-indexed: in femtoseconds
    df.index = np.round(df.index*1000).astype(int)
    ts = round(min(np.diff(df.index)))
        
    ## reindex to bring back missing times
    df = df.reindex(np.arange(df.index.min(),df.index.max()+1, ts).astype(int))
    
    df = df.apply(lambda col : gaussian_filter1d(col, halflife / ts, truncate=3))
    
    df.index = df.index.astype(float) / 1000
    
    return df # .dropna()
    
# =============================================================================
# %% load a Gs (van Hove) file quickly. This is here to avoid bloat in the 
# ## macroscopic analysis notebook
# =============================================================================

def load_gs(glob_query, option, **kwargs):
    
    gs_glob = glob(glob_query)
    
    gs = None
    try: 
        gs = pd.read_csv(gs_glob[0])
        # if option != 'spectra' : 
        gs.gs = gs.gs * gs.r**2
        gs = gs.set_index(['r','time']).unstack().apply(lambda col: col/col.sum(), axis=0)
        gs.columns = [x[1] for x in gs.columns]
        gs.index = np.round(gs.index.values,4)
    except:
        print(f'could not load a Gs file for {glob_query}')
        
    ## return times at which Gs decays to 1/e
    if option in ['a2', 'r2', 'exponent', 'cds', 'exp-vs-r', 'cdt', 'Funke'] and gs is not None:
        if 'radii' in kwargs.keys() :
            decay_times = list()
            for r in kwargs['radii'] :
    
                rmin=min(r); rmax=max(r)
                
                s = gs.loc[rmin:rmax,:].sum().reset_index()
                s.columns = ['time','gs']
                # s.set_index('time', inplace=True)
                
                ## reverse interpolation: x-values are the function, y is time
                gs_int = interp1d(s.gs, s.time)
                
                decay_times.append(gs_int(1/np.e))
            return decay_times
                
        else : print('to calculate 1/e times, supply an iterable of tuples, "radii".')
    
    return gs

# =============================================================================
# %% 4-point correlation - for Burnett and response functions
# ## assumes a regularly spaced, real-valued zero-mean series, e.g. velocities
# ## assumes time lags are measured in periods of the series, not in real units. 
# =============================================================================

def four_point_autocorr(series, lag1, lag2, lag3):
    '''
    Compute a four-point (three-lag) correlation of a series with itself for 
    three lags. The series is assumed to be real-valued (complex conjugation is
    ignored) and zero-mean. 

    Parameters
    ----------
    series : 1D series such as a numpy array
        The series for which the autocorrelation is computed. Assumed to be 
        real-valued and zero-mean.
    lag1 : int
        First lag (in sequence of 3), an integer number of indices of the series.
    lag2 : int
        Second lag (in sequence of 3), an integer number of indices of the series.
    lag3 : int
        Third lag (last in sequence of 3), an integer number of indices of the series.

    Returns
    -------
    float
        The four-point (three-lag) autocorrelation of the series.
        
    Throws
    -------
    AssertionError
        Throws an AssertionError if the sum of lags is larger than the length 
        of the series.

    '''
    
    assert lag1 + lag2 + lag3 <= len(series), 'sum of 4-pt correlation time lags is larger than series is long'
    
    return np.mean(series[:-lag1-lag2-lag3 if lag1+lag2+lag3 else None,:] * 
                  series[lag1:-lag2-lag3 if lag2+lag3 else None,:] * 
                  series[lag1+lag2:-lag3 if lag3 else None,:] * 
                  series[lag1+lag2+lag3:,:], axis=0)

# =============================================================================
# %% 4th cumulant - for Burnett and response functions
# ## assumes a regularly spaced, real-valued zero-mean time series, e.g. velocities
# ## assumes time lags are measured in periods of the series, not in real units. 
# =============================================================================

def fourth_cumulant(series, lag1, lag2, lag3):
    '''
    Compute the fourth cumulant of a series, e.g. of a velocity series in time. 
    The series is assumed to be regularly-spaced, zero-mean, real-valued.

    Parameters
    ----------
    series : 1D series such as a numpy array
        The series for which the cumulant is computed. Assumed to be regularly
        spaced, real-valued, and zero-mean.
    lag1 : int
        First lag (in sequence of 3), an integer number of indices of the series.
    lag2 : int
        Second lag (in sequence of 3), an integer number of indices of the series.
    lag3 : int
        Third lag (last in sequence of 3), an integer number of indices of the series.

    Returns
    -------
    float
        Fourth cumulant of the series.

    '''
    
    autocorr_4 = four_point_autocorr(series, lag1, lag2, lag3)
    
    series_tau_1 = series[:-lag1-lag2-lag3 if lag1+lag2+lag3 else None,:]
    series_tau_2 = series[lag1:-lag2-lag3 if lag2+lag3 else None,:]
    series_tau_3 = series[lag1+lag2:-lag3 if lag3 else None,:]
    series_tau_4 = series[lag1+lag2+lag3:,:]
    
    prod_12 = np.mean(series_tau_1 * series_tau_2, axis=0)
    prod_13 = np.mean(series_tau_1 * series_tau_3, axis=0)
    prod_14 = np.mean(series_tau_1 * series_tau_4, axis=0)
    
    prod_23 = np.mean(series_tau_2 * series_tau_3, axis=0)
    prod_24 = np.mean(series_tau_2 * series_tau_4, axis=0)
    
    prod_34 = np.mean(series_tau_3 * series_tau_4, axis=0)
    
    return autocorr_4 - prod_12*prod_34 - prod_13*prod_24 - prod_14*prod_23

# =============================================================================
# %% compute Burnett CF from velocities by taking 4th cumulant and integrating
# =============================================================================

def burnett_from_velocity(velocity_series, total_lag, downsample = 0):
    '''
    Compute Burnett CF for series at one lag.
    Sources: 
        Nieuwenhuizen & Ernst, J. Stat. Phys. (1985) vol. 41, p. 773
        Song et al., PNAS (2019) vol. 116, p. 12733

    Parameters
    ----------
    velocity_series : series such as 1D numpy array or pandas series
        Time series of velocities, assumed: regularly spaced, real-valued, zero-mean.
    total_lag : int
        Time lag, an integer number of indices/periods of the series. 
        This is the sum of 3 lags over which the 4-pt CF is computed.
    downsample : int
        Take every downsample-th set of lags if total_lag > downsample * 10.
        If not zero, then speeds up calculation by downsample**2 times.

    Returns
    -------
    bcf : float
        Burnett CF at the value of time lag. 
        Units: [(velocity series distance unit)^4/(velocity series time point spacing)^2].
        As of 2021/02/25, diagonal terms.

    '''
    
    if total_lag == 0 : return fourth_cumulant(velocity_series, 0, 0, 0)
    
    if (not downsample) or total_lag/downsample < 10 :
        downsample = 1
    
    ## add the 4th cumulant for each combination of lag1 and 
    ## the last lag is determined by the first two, hence double sum, not triple
    bcf = 0
    samples = 0
    for lag1 in np.arange(0,total_lag+1, downsample, dtype=int) :
        for lag2 in np.arange(0,total_lag+1-lag1, downsample, dtype=int) :
            bcf += fourth_cumulant(velocity_series, lag1, lag2, total_lag-lag1-lag2)
            samples += 1
            
    ## since each pathway is added, the total value 
    total_samples = sum(range(len(np.arange(0,total_lag+1))+1))
    
    return bcf * total_samples / samples

# =============================================================================
# %% shorthand for loading macro planes
# =============================================================================

def load_macro_planes(planes_to_load, load_r2=False, load_com=False) :
    
    ## make a structure for loading data
    planes_dicts = []
    
    for plane in planes_to_load.itertuples(index=False):
    
        mm = plane.metal
        T1 = plane.T1
        hp = plane.hop_path
        ph = plane.phase
        st = plane.stoich
        ex = plane.exclude
        # tt = plane.total_time ## based on hops, not CoM. Could be off.
        cn = plane.config
        
        ## load lammps structure
        _, _, cell, atoms = cu.read_lmp(plane.lammps_path, fractional=False)
        a2_folder = '/'.join(plane.lammps_path.split('/')[:-1])
        
        ## load lammps r2 file for the diffusion coefficient
        if load_r2 :
            r2_fname = glob.glob(a2_folder+f'/a2_*{T1}K-{mm}.fix')
            
            ## load the r2 file if exactly one exists, else complain
            if isinstance(r2_fname, list) and len(r2_fname) == 1:
                
                ## read the r2 file - options for fix file
                this_r2 = pd.read_csv(r2_fname[0], names=['time','r2','r4','a2'],
                                      skiprows=2, sep=' ')
                this_r2.time /= 1000
                
                this_r2.set_index('time', inplace=True)
                
                ## Look for a literature folder
                lit_folder = '/'.join(a2_folder.split('/')[:-1])        
                
                print(f'\nLoaded r2 for plane {hp}')
            else:
                print(f'\nsomething off with plane {hp}.')
                print(f'here are possible r2 outputs: {r2_fname}')
                this_r2 = None
        else : this_r2 = None
        
        ## the a2 fix file is LAMMPS output, csv is calculated with multiple starts
        ## this takes the longest-duration a2 file
        a2_fnames = glob(a2_folder+f'/{mm}*a2-*{T1}K*ps.csv')
        
        ## load the a2 file if exactly one exists, else complain
        if a2_fnames :
            if len(a2_fnames) > 1 : a2_fnames = sorted(a2_fnames, reverse=True,
                            key = lambda x : eval(split('-|_| ',x)[-1][:-6]))
            
            ## read the a2 file - options for csv file
            this_a2 = pd.read_csv(a2_fnames[0], sep=',').set_index('time')
            
            ## Look for a literature folder
            lit_folder = '/'.join(a2_folder.split('/')[:-1])        
            
            print(f'Loaded a2: {a2_fnames[0]}')
        else:
            print(f'something off with plane {hp}.')
            print(f'here are possible a2 outputs: {a2_fnames}')
            this_a2 = None
            
        ## load the CoM trajectory if it exists
        com_fname = glob(a2_folder + f'/cm*{T1}K*{mm}.fix')
        if isinstance(com_fname, list) and len(com_fname) == 1 and load_com:
            
            this_com = pd.read_csv(com_fname[0],sep=' ', names=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'], skiprows=2).drop(columns=['vx','vy','vz'])
            this_com.time /= 1000. ## hard-coded conversion from steps to picoseconds
            this_com.set_index('time', inplace=True)
            print('Loaded CoM trajectory.')
        elif not load_com :
            this_com = True
            # print('Skipping CoM trajectory.')
        else : 
            print(f'Could not load CoM trajectory, found: {com_fname}')
            this_com = None
           
        ## wrap the a2, CoM, and metadata into a dict
        if (this_r2 is not None or not load_r2) and (this_a2 is not None)  :
            planes_dicts.append(dict(phase=ph, metal=mm, T1=T1, config=cn, stoich=st, exclude=ex,
                                     a2=this_a2, lit_folder=lit_folder, com = this_com,
                                     cell=cell, atoms=atoms, folder=a2_folder, r2=this_r2))
            
    ## make the holding structure into a dataframe    
    return pd.DataFrame(planes_dicts)

# =============================================================================
# %% Timescales of trapping by oxygen defects in beta-aluminas
# =============================================================================

def trapping_times(hops, verbose = False) :
    ## output all the times all ions spend within r≤1 of a defect
    
    ## track 
    hops['future_ox_r'] = np.nan
    hops.future_ox_r.iloc[:-1] = hops.new_ox_r.iloc[1:].values
    
    trap_times = list()
    
    for _, g in hops.query('rev_hop == False & new_ox_r == 2').groupby('ion'):
        
        trapping = g.query('old_ox_r != 1 & future_ox_r == 1')
        detrapping = g.query('old_ox_r == 1 & future_ox_r != 1')
        
        ## account for ion starting or finishing in a trap
        if len(trapping) > len(detrapping) :
            trapping = trapping.iloc[:-1]
        elif len(trapping) < len(detrapping) :
            detrapping = detrapping.iloc[1:]
        
        trap_times.append(list(abs(detrapping.time.values - trapping.time.values)))
        
    return list(flatten(trap_times))

        
        























