#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andreypoletaev
"""

# =============================================================================
# %% Imports
# =============================================================================

import sys, os, random, freud
if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import numpy  as np
import networkx as nx
import crystal_utils as cu

from matplotlib import pyplot as plt

# =============================================================================
# %% Test the neighbor-getting method with real-space coordinates.
# ## random seed for Na102: 10
# =============================================================================
## starting input .CONFIG file with atoms
prod_file = './templates/stNabeta_ortho10102.CONFIG'

## parameters and constants
metal = 'Na'

random.seed(15)  ## Set the random seed. 
exclude = 4  ## minimum network distance between mid-oxygens to enforce when picking where to add interstitials

## Example 1.20 stoichiometry : this is the # of interstitials in every plane
## (the code enables varying the number for every plane as well)
num_ois_per_plane = 10
packed = False

## calculate stoichiometry of the resulting phase for naming the output files below.
stoich = 2 * num_ois_per_plane + 100

to_dir = f'./{metal} {stoich}_{exclude}/' ## directory to which grids, files & distances are written
if not os.path.exists(to_dir): os.makedirs(to_dir)

## read the input .CONFIG file
phase, intro_line, cell, atoms = cu.read_poly(prod_file, fractional=False)

## find all conduction planes in the input file
planes = cu.get_conduction_planes(atoms, metal)

## initialize data structures
new_atoms = atoms.copy(deep=True)
all_interstitial_indices = list()
all_paths_to_oi = list() ## plotting only

## for every plane, find mid-oxygen sites
for pl in planes:

    ## get the mobile-ion sites for this plane
    mobile_sites = cu.get_mobile_ion_sites(atoms, pl, cell)
    ## get all the mid-oxygen sites in this plane
    mid_oxs, edges, midpts = cu.get_mid_oxygen_sites_freud(mobile_sites, cell, viz=True)
    plt.gca().set(aspect=1, title=f'z={pl:.3f} $\AA$', ylim=np.array([-0.59,0.59])*cell[1,1],
                  xlim=np.array([-0.55,0.55])*cell[0,0])
    plt.gcf().tight_layout()

    ## Pick mid-oxygen sites quasi-randomly, independent of coordinates
    if packed: 
        picked, fresh, past = cu.generate_mid_oxygens_packed(mid_oxs, num_ois_per_plane, exclude)
    else:
        picked, fresh, past = cu.generate_mid_oxygens(mid_oxs, num_ois_per_plane, exclude)
    print(f'{len(picked)} picked, {len(past)} excluded, {len(fresh)} sites remain')

    ## Plot the mid-oxygen sites where interstitials are going
    xs = []; ys = []
    for mo in mid_oxs:
        for p4 in picked:
            if mo[1] == p4:
                xs.append(mo[0][0]);
                ys.append(mo[0][1])
    plt.gca().scatter(xs, ys, c='tab:green', s=100, label=f'{num_ois_per_plane} mO\'s, {exclude}+ apart')

    ## save the sites as oxygen sites for later
    metal_sites_next_to_oi = np.array(sorted([x for x in cu.flatten(list(picked))]))

    pl_int = int((pl/cell[2,2] + 0.5) * 100)  ## shifting from (-L/2, L/2) to (0,L)
    # np.savetxt(to_dir+f'oxygen_cells_{pl_int:03d}.csv', metal_sites_next_to_oi,fmt='%d')

    ## take the picked site edges back to xyz coordinates
    mid_oxs_to_add = [x for x in zip(xs, ys, np.ones(len(xs)) * pl)]

    ## plot mid-oxygen sites that are excluded as close to will-be interstitials
    xs_e = []; ys_e = []
    for mo in mid_oxs:
        for p3 in past - picked:
            if mo[1] == p3:
                xs_e.append(mo[0][0]);
                ys_e.append(mo[0][1])
    plt.gca().scatter(xs_e, ys_e, c='k', s=30, label=f'mO\'s 1-{exclude} away')

    ## pretty plot thing
    plt.gca().legend(ncol=2)

    ## actually add the defects and mobile ions to the pandas dataframe
    ## keep track of what indices the interstitial oxygens have
    new_atoms, oi_indices = cu.create_roth_defects(new_atoms, pl, mid_oxs_to_add, cell, verbose=False)

    ## display / output the indices of interstitial oxygen atoms
    # print('indices for interstitials:', oi_indices)
    all_interstitial_indices.append(oi_indices)

    ## create a proper networkx graph from edge list, and calculate path lengths
    nxg = nx.from_edgelist(edges)
    path_lengths = cu.path_lengths(nxg)

    ## measure all path lengths to the oxygens; this yields a list
    all_paths_to_oi.append([min(path_lengths[metal_sites_next_to_oi,x]) for x in range(len(mobile_sites))])
    
    ## save figure for posterity
    plt.savefig(to_dir+f'{metal}beta{stoich}_{exclude}_{pl_int:03d}_picking.png', dpi=150)
    
## write the DL_POLY .CONFIG file - but it is not actually useful
cu.write_poly(to_dir+f'{metal}beta{stoich}_{exclude}.CONFIG',phase, intro_line, cell, new_atoms, fractional=False)

## remove nesting in the list of indices of interstitial oxygens
all_interstitial_indices = [x for x in cu.flatten(all_interstitial_indices)]

## write the LAMMPS file
cu.write_lmp(to_dir+f'{metal}beta{stoich}_{exclude}.lmp', cell, new_atoms, all_interstitial_indices,fractional=False)

## remove nesting in in the path lengths to oxygen sites
all_paths_to_oi = np.array([x for x in cu.flatten(all_paths_to_oi)])

## plot a histogram of the distances from mobile-ion sites to O interstitials
fig, ax = plt.subplots()
ax.hist(all_paths_to_oi, bins=np.arange(max(all_paths_to_oi)+2)-0.5,edgecolor='k',linewidth=0.5);
ax.set(title=f'{stoich} excluding {exclude}');
fig.tight_layout();

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
to_dir = f'./{metal} {stoich}_{exclude}/'

## try importing a python-generated .lmp file
fn = to_dir + f'{metal}beta{stoich}_{exclude}.lmp'  ## this should be carried over from above
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


