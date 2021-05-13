#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:34:59 2020

@author: andreypoletaev
@editor: avrumnoor
"""

import numpy as np
import pandas as pd

# Turn off SettingWithCopyWarning (default='warn')
# (See stackoverflow.com/questions/20625582/)
pd.options.mode.chained_assignment = None

import freud
import freud.box

import re

from scipy.spatial import distance as ssd
from scipy.spatial import Voronoi

import matplotlib as mpl
from matplotlib import pyplot as plt

from datetime import datetime as dt

import random
import itertools
import networkx as nx

from collections import Counter

## dictionaries for shorthands later on
## atom masses in LAMMPS structure files: 'masses' is invoked when reading a file
masses = {16: 'O', 23: 'Na', 39: 'K', 108: 'Ag', 27: 'Al', 7: 'Li', 24: 'Mg'}

## 'write_masses' is invoked when writing atom masses to a (newly being created) LAMMPS structure file
write_masses = {'O': 15.999, 'Na': 22.98976928, 'Al': 26.9815386,
                'Ag': 107.8682, 'K': 39.0983, 'Mg': 24.305, 'Li': 6.941}

## 'dims' stands for dimensions, and serves as a way to avoid remembering array indices
## when converting between physical and fractional units and reading/writing cell sizes
dims = {0: 'x', 1: 'y', 2: 'z'}

## ionic charges for specific atoms in our structures. Used when writing LAMMPS structure files
charges = {'Al': 3, 'O': -2, 'Na': 1, 'Li': 1, 'K': 1, 'Ag': 1, 'Mg': 2}

## order of atoms in LMP files
atom_kind_order = {'Al': 2, 'O': 3, 'Na': 1, 'Li': 4, 'K': 1, 'Ag': 1, 'Mg': 4, 'Oi' : 4}

# =============================================================================
# %% a list flattening function for lists of strings (filenames)
# ## flatten returns an iterator (usually sufficient), 
# ## flattened makes it into a proper list
# ## NOTE: this will throw an error if the list has some one-level items, e.g.
# ## example: this will work: flatten([[1,2],[3,4]])
# ## example: this will error: flatten([1,2,[3,4]])
# =============================================================================

flatten = lambda l: itertools.chain.from_iterable(itertools.repeat(x, 1) if isinstance(x, str) else x for x in l)


# =============================================================================
# %% check whether a point is in a rectangle bound by (x,y,...)_min & (...)_max
# ## For floating-point arithmetic, equality is allowed as of 2020/06/12.
# =============================================================================

def in_bounds(pt, xymin, xymax):
    ''' check if point 'pt' is between (xy)_min and (xy)_max in 2D, 3D or more
        2020/06/13 : added left side equality 
        2020/07/08 : realized this is a general method for any # of dimensions
        '''

    return (sum(pt >= xymin) == len(pt)) & (sum(pt <= xymax) == len(pt))

# =============================================================================
# %% return standard plane name for one z-coordinate
# ## wnat many? run list comprenension
# =============================================================================

def standard_plane_name(fractional_z):
    # if len(fractional_zs) > 1 : return [f'{int((z+0.5)*100):03d}' for z in fractional_zs]
    return f'{int((fractional_z+0.5)*100):03d}'

# =============================================================================
# %% read a DL_POLY .CONFIG file
# =============================================================================

def read_poly(filename, fractional=True):
    ''' reads a DL_POLY .CONFIG file to a pandas dataframe.
        No empty lines between anything in the input! 
	Only tested on orthogonal cells.

        'fractional' flag: 
	whether to divide the xyz positions of the atoms by cell dimensions 
	in the output. By default, this is True. 
	
	outputs: 
	'phase' = 1st line of the input file, usually has the overall counts of atoms,
	is returned without modification.
	'line2' = second line of the input file, I do not know what it is for. 
	'cell' = dimensions of the total cell in the input file, np.ndarray with shape (3,3) 
	'atoms' = the pandas dataframe with all the atoms in the file. Index is 
	called 'idx', columns are 'atom' (strings), and float 'x', 'y', 'z' for positions. 
	'''

    ## initialize cell dimensions
    cell = np.zeros((3, 3))

    ## initialize lists for actual data; lists preserve order for later writing
    atom_names, atom_indices, atom_x, atom_y, atom_z = [[], [], [], [], []]

    ## open the input file and read it line by line
    with open(filename) as fin:

        phase = fin.readline()  ## read one header line, overall chem composition
        line2 = fin.readline()  ## read 2nd header line, unsure what it does
        for r in range(3):  ## read cell dimensions
            cell[r] = [float(i) for i in re.split(' |\t', fin.readline()[:-1]) if i]

        ## read atoms until all done and out of atoms
        line = fin.readline()  ## read the first line, this should be an atom name and index
        while line:
            atom_name, atom_index = [i for i in re.split(' |\t',line[:-1])  if i]
            atom_names.append(atom_name)
            atom_indices.append(int(atom_index))
            line = fin.readline()  ## read line with positions
            x, y, z = [float(i) for i in re.split(' |\t',line[:-1]) if i]
            atom_x.append(x)
            atom_y.append(y)
            atom_z.append(z)
            line = fin.readline()

    ## construct the atoms dataframe
    atoms = pd.DataFrame(data={'atom': atom_names, 'idx': atom_indices,
                               'x': atom_x, 'y': atom_y, 'z': atom_z})

    ## Set the index of the dataframe to the meaningful index of the atoms from 
    ## from the original input file rather than the arbitrary ordinal numbers 
    ## that pandas assigns upon construction.
    atoms.set_index('idx', inplace=True)

    ## Make fractional coordinates: divide the physical ones that were read in
    ## by the cell size. Note that for non-orthogonal cells this would be
    ## a slightly more involved linear transformation because of the tilt angles.
    ## That is not currently implemented.
    if fractional:
        atoms.x /= cell[0, 0]
        atoms.y /= cell[1, 1]
        atoms.z /= cell[2, 2]
        
    ## adjust coordinates to make them go -L/2 to L/2 in case they are 0 to L
    for i, d in dims.items():
        if 0.9 * cell[i, i] < atoms[d].max(): atoms[d] -= 0.5 * cell[i, i]

    return phase, line2, cell, atoms

# =============================================================================
# %% write a completed / modified structure to a new .CONFIG file! 
# =============================================================================

def write_poly(filename, phase, intro_line, cell, atoms, fractional=True):
    ''' Writes a DL_POLY .CONFIG file using atoms.
        If a file with this name exists, it will be overwritten. 
	ONLY orthogonal cells.
	No longer assumes fractional coordinates (dimensionless, -0.5 to 0.5 or 0 to 1)
    Assumes that the index of the 'atoms' dataframe is the index of the atoms. 
	'''
    
    ## adjust coordinates to make them go -L/2 to L/2 in case they are 0 to L
    for i, d in dims.items():
        if 0.9 * cell[i, i] < atoms[d].max(): atoms[d] -= 0.5 * cell[i, i]

    ## Open an output file. This will overwrite if a file with this name exists.
    fout = open(filename, 'w')

    ## Write the first two lines for the .CONFIG file. Could modify the first line
    ## to reflect the total counts of atoms - not implemented.
    fout.write(phase)
    fout.write(intro_line)

    ## write the dimensions of the simulation box
    for i in range(3):
        fout.write('\t' + '\t'.join([f'{x:.5f}' for x in cell[i].tolist()]) + '\n')

    ## some form of progress / output / "I'm OK" statement
    print(f'writing {len(atoms)} atoms')

    ## write atoms one by one, breaking each up into two lines in the .CONFIG
    ## convert back to physical dimensions based on the provided simulation 
    ## size in the input 'cell'. Note: this is only implemented for orthogonal 
    ## cells, and will not work for non-orthogonal ones.
    for atom in atoms.iterrows():

        ## write the atom identity and ordinal index
        fout.write(atom[1].atom + '\t' + str(int(atom[0])) + '\n')

        ## convert to physical coordinates from fractional
        if fractional:
            x = atom[1].x * cell[0, 0]
            y = atom[1].y * cell[1, 1]
            z = atom[1].z * cell[2, 2]
        else:
            x, y, z = [atom[1].x, atom[1].y, atom[1].z]

        ## write the coordinates
        fout.write(f'   {x:.6f}\t{y:.6f}\t{z:.6f}\n')

    ## close the output file.
    fout.close()

# =============================================================================
# %% read a LAMMPS .lmp structure file with atoms
# =============================================================================

def read_lmp(filename, ignore=[], fractional=True):
    '''
    Reads a LAMMPS .lmp file with a crystal structure, and outputs pandas. Optional
    to exclude any particular 'types' of atoms. Restrictions: only orthogonal 
    simulation boxes, does not deal with simulation boxes that have tilt. 
    READ THE DESCRIPTION OF THE OUTPUT DATAFRAME!

    Parameters
    ----------
    filename : string
        Path to the file that is to be read.
    ignore : list, optional
        These 'types' of atoms will not be returned. The default is an empty list.
    fractional : bool, optional
        Output fractional coordinates scaled to the range (-0.5, 0.5) if True. 
        If false, will output real-space coordinates from the LAMMPS file. 
        The default is True.

    Returns
    -------
    phase : string
        First line of the file, typically chemical formula for the contents.
    line2 : string
        Weird second line of the LAMMPS file, + for conversion with DL_POLY.
    cell : numpy array, shape (3,3)
        Dimensions of the simulation box in units for LAMMPS (typically angstrom).
    atoms : pandas DataFrame
        pandas DataFrame with rows being the int indices of atoms, and columns:
            'atom', 'x', 'y', 'z', 'type'. The 'atom' column is string with the
            chemical symbol, 'type' is an integer from the lammps file. So if 
            there are, for example, multiple unique 'types' for atom=='O', then
            this file has interstitial oxygens in a separate type. 
            Columns 'x', 'y', 'z' are coordinates. The coordinates are centered 
            at zero, i.e. are in the range (-L/2, +L/2) rather than (0, L).

    '''
    
    try:
        ## open the file to read
        with open(filename) as fin:
    
            ## name of the phase
            phase = fin.readline()[:-1]  ## this will have a # comment sign in front
            fin.readline()  ## second line is empty
    
            ## atom counts and types
            atomcount = int([i for i in fin.readline()[:-1].split(' ') if i][0])  ## ' NNNN atoms'
            atomtypes = int([i for i in fin.readline()[:-1].split(' ') if i][0])  ## ' T atom types'
            fin.readline()  ## empty line
    
            # print(f'Found {atomcount} atoms of {atomtypes} types, ignoring {len(ignore)} types')
    
            ## cell dimensions
            cell = np.zeros((3, 3))
            lmp_dims = {'xlo': 0, 'ylo': 1, 'zlo': 2, 'xhi': 0, 'yhi': 1, 'zhi': 2}
            for i in range(3):  ## read cell dimensions
                contents = [i for i in fin.readline()[:-1].split(' ') if i]
                cell[lmp_dims[contents[3]], lmp_dims[contents[3]]] = float(contents[1]) - float(contents[0])
            fin.readline()  ## this is the empty line after cell
    
            ## masses
            fin.readline()  ## 'Masses'
            fin.readline()
    
            ## read atom masses and assign chemical identities
            atomkinds = dict()
            for i in range(atomtypes):  ## atom type and mass. Getting identities here
                contents = [i for i in fin.readline()[:-1].split(' ') if i]
                # print(contents)
                atomkinds[int(contents[0])] = masses[int(round(float(contents[1]), 0))]
    
            line = fin.readline()  ## some number of empty lines
            while line:
                if 'Atoms' in line: break  ## could deal with atom style here if given
                line = fin.readline()
    
            fin.readline()  ## empty line
    
            ## actually read atoms to pandas dataframe
            rows_list = []
            line = fin.readline()  ## first atom
            while line:
                row = [i for i in line[:-1].split(' ') if i]
                atom_type = int(row[1])
                if atom_type not in ignore:  ## helps to deal with interstitials
                    rowdict = {'idx': int(row[0]), 'atom': atomkinds[atom_type],
                               'x': float(row[-3]), 'y': float(row[-2]), 'z': float(row[-1]),
                               'type':atom_type}
                    rows_list.append(rowdict)
                ## read one more atom 
                line = fin.readline()
    
            ## make a dataframe
            atoms = pd.DataFrame(rows_list)
            atoms.set_index('idx', inplace=True)
    
            ## center at zero, and return fractional coordinates
            for i, d in dims.items():
                atoms[d] -= 0.5 * cell[i, i]
                if fractional: atoms[d] /= cell[i, i]
    
            ## output something about it going well
            # print(f'Done reading {len(atoms)} atoms')
    
            ## make a header line for later possibly saving as .CONFIG
            line2 = '\t0\t{:d}\t{}\n'.format(len(set(atomkinds.values())), atomcount)
            
            return phase, line2, cell, atoms
            
    except IOError : return f'could not find this file: {filename}'

# =============================================================================
# %% write a LAMMPS file, adding an extra category for interstitial oxygens 
# ## Some lines here look weird: this preserves the formatting in the output
# ## Does it matter? I don't want to deal with whether it does. 
# ## Plus, if the output .lmp file gets read later by the above read_lmp(), 
# ## I would rather not worry about an extra character to take out everywhere.
# =============================================================================

def write_lmp(filepath, cell, atoms, defects=None, fractional=True, defect='O'):
    ''' write a LAMMPS input file, making an extra category for Oxygen interstitials
    
        input filepath: this will be the location of the output
        input cell:     dimensions of the simulation box, assumed orthogonal
        input atoms:    pandas dataframe with atoms, assumed fractional coordinates
        input defects:  integer indices of Mg/Oi-to-be (a new atom type will be created)
        input fractional: '''

    ## copy the dataframe so it does not get modified in place
    atoms_to_write = atoms.copy(deep=True)
    there_are_defects = (defects is not None and len(defects) > 0)

    ## scale to real dimensions from fractional dimensions
    # for i, d in dims.iteritems(): atoms_to_write[d] *= cell[i, i] ## this was python2
    for i, d in dims.items():
        if fractional: atoms_to_write[d] *= cell[i, i]
        ## adjust coordinates to make them go 0 to L in case they are -L/2 to L/2
        if -0.9 * cell[i, i] < atoms_to_write[d].min() < 0: atoms_to_write[d] += 0.5 * cell[i, i]
        
    # ## adjust coordinates to make them go 0 to L in case they are -L/2 to L/2
    # if -0.9 < atoms_to_write.x.min() < 0: atoms_to_write.x += 0.5
    # if -0.9 < atoms_to_write.y.min() < 0: atoms_to_write.y += 0.5
    # if -0.9 < atoms_to_write.z.min() < 0: atoms_to_write.z += 0.5

    ## write intro line
    fout = open(filepath, 'w')
    fout.write(' # generated in python, ' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')

    ## keep track of how many kinds of atoms there are
    atom_kinds = atoms_to_write.atom.unique()
    atom_kinds = np.array(sorted(list(atom_kinds), key=lambda x : atom_kind_order[x]))
    print('order of atoms: ', atom_kinds)
    atom_dict = dict()

    ## write #'s of atoms and # of types of atoms 
    fout.write('{:>12}'.format(len(atoms_to_write)) + '  atoms\n')
    fout.write('{:>12}'.format(int(len(atom_kinds) + there_are_defects)) + '  atom types\n\n')

    ## write cell dimensions
    for i in range(3):
        fout.write('      ' + '0.00000000      ' + f'{cell[i, i]:.8f}' + \
                   f'  {dims[i]}lo {dims[i]}hi\n')
    fout.write('\n')

    ## write atom kinds and their masses
    fout.write('Masses\n\n')

    ## assign indices to atom types and write them
    ## simultaneously mutate the dataframe
    ## doing this in this way ensures that the defect is the last kind of atom
    for i, a in enumerate(atom_kinds):
        atom_dict[i + 1] = a  ## LAMMPS starts indexing at 1 when python starts at 0
        atoms_to_write.loc[atoms_to_write.atom == a, 'atom'] = i + 1  ## substitution
        fout.write(f'           {i+1:d}   {write_masses[a]:.8f}    # {a}\n')
    if there_are_defects:
        fout.write(
            f'           {len(atom_kinds)+1:d}   {write_masses[defect]:.8f}    # {defect} defect\n')
        atom_dict[len(atom_kinds)+1] = defect

    fout.write('\n\n')

    ## write the header for the atoms section
    fout.write('Atoms # charge\n\n')

    ## write atoms one by one
    for a in atoms_to_write.iterrows():
        idx = int(a[0]);
        kind = a[1].atom;

        ## substitute defect atom type for a new one
        if there_are_defects and (idx in defects):
            kind = len(atom_kinds) + 1
        
        charge = charges[atom_dict[kind]]

        ## write the atom
        ## potential discrepancies with atomsk-generated file: (1) extra space with negative charge
        ## (2) spaces in front of coordinates are constant here at 7. 
        ## This spacing seems to work (2020/06/30)
        fout.write('     {:5d}    {:.0f}   {:.6f}       {:.8f}       {:.8f}       {:.8f}\n'.format(
            idx, kind, charge, np.round(a[1].x, 5), np.round(a[1].y, 5), np.round(a[1].z, 5)))

    fout.close()

    print(f'\nDone writing file {filepath}\n')

# =============================================================================
# %% get conduction planes by binning the z coordinates of mobile ions
# ## TODO: remove duplicates from here if they exist
# =============================================================================

def get_conduction_planes(atoms, mobile_ion, inexact=False):
    ''' output z-coordinates of conduction planes 
        input atoms: pandas database with atom names and x,y,z coordinates
        input mobile_ion: string, must match the 'atom' column of atoms
        input inexact: for beta-doubleprime, merge pairs of nearby indices into one '''

    zs = sorted(atoms.query('atom == @mobile_ion').z.unique().tolist())
    
    if not inexact : return np.array(zs)
    else : 
        thresh = (max(zs) - min(zs)) / (len(zs)-2) * 2 / 11 * 0.2
        return [np.mean([zs[i], zs[i+1]]) for i in range(len(zs)-1) if np.diff(zs)[i] < thresh]

# =============================================================================
# %% remove "duplicate" conduction planes. This is done by
# ## merging the pairs of planes generated by the
# ## get_conduction_planes(atoms, mobile_ion) function that are close within
# ## certain margin.
# ## Input: Array of planes, threshold: 0.003 (most likely)
# ## Output: Array of planes
# =============================================================================

def remove_duplicate_planes(planes):
    avg_planes = []
    planes = list(planes)
    planes.sort()
    for i in range(0, len(planes), 2):
        average = (planes[i] + planes[i + 1]) / 2
        avg_planes.append(average)
    avg_planes = np.array(avg_planes)  # avg_planes = average of the two pairs of z-coordinates of the planes
    return avg_planes

# =============================================================================
# %% get mobile ion sites by creating a Voronoi tessellation wrapped around the
# ## periodic boundary conditions by starting with the non-interstitial oxygens
# ## Note: this assumes that only the non-interstitial oxygens are in the 
# ## conduction plane by z; an input with interstitials will yield a mess. 
# ## That is why the plotting is an option: it is a fast visual way to debug.
# =============================================================================

def get_mobile_ion_sites(atoms, plane, cell=np.eye(3), viz=False, thresh=0.03):
    '''
    Outputs (x,y,plane) coordinates for all mobile ion sites in plane. Cell is 
    assumed orthogonal. Uses the fact that non-interstitial oxygens are at the 
    z=plane for both beta- and beta"-aluminas.

    Parameters
    ----------
    atoms : pandas DataFrame
        atom names, indices, and x,y,z coordinates.
    plane : number (float)
        z-coordinate of the conduction plane, assumed parallel to xy.
    cell : numpy array of shape (3,3), optional
        Dimensions of the simulation box, assumed orthogonal. The default is 
        np.eye(3) for the case that 'atoms' is in fractional coordinates.
    viz : boolean, optional
        Flag on plotting a 2D rendering of the result. The default is False.
    thresh : float, optional
        z-coordinate threshold for finding in-plane oxygens. Same units as cell.

    Returns
    -------
    mobile_sites : numpy array of shape (n,3)
        Coordinates of points at centers of mobile-ion sites in this plane.

    '''

    ## warn that lists will no longer work
    if (type(plane) == list):
        print('As of 2020/07/08, get_mobile_ion_sites() only works on single planes, not lists')
        print(f'z={plane[-1]:.4f} will be returned.')
        plane = plane[-1]

    these_sites = list()

    if viz: print(f'z = {plane:.3f}')
    ## find all (non-interstitial) oxygens
    oxygens = atoms.query(f'atom == "O" & @plane - {thresh} < z < @plane + {thresh}')

    ## make a Voronoi object with existing oxygens as centers
    gridpts = oxygens[['x', 'y', 'z']].values

    ## this might need flattening to 2D
    box = freud.box.Box(Lx=cell[0, 0], Ly=cell[1, 1], is2D=True)  ## used to be squareL=1
    gridpts = box.wrap(gridpts)

    ## this makes hex polys w/ Ox as centers; 2nd param is the overshoot for periodicity
    site_vor = freud.locality.Voronoi(box)
    site_vor.compute(system=(box, gridpts))

    ## make a plot
    if viz:
        fig, ax = plt.subplots()
        draw_voronoi(box, gridpts, site_vor.polytopes, draw_points=True, ax=ax, draw_box=True)

    # round the cell dimensions because the points will be rounded too
    Lx = np.round(cell[0, 0],4)
    Ly = np.round(cell[1, 1],4)
    out_of_bounds = 0
    for poly in site_vor.polytopes:
        for vertex in poly:
            if in_bounds(vertex[:2], -0.5*np.array([Lx, Ly]), 0.5*np.array([Lx, Ly])) :
                these_sites.append([vertex[0], vertex[1], plane])  ## append avg z given that z was previously lost
            ## check for the corner piece especially
            elif sum(np.isclose(abs(vertex[:2]), np.array([0.5*Lx, 0.5*Ly]), rtol=1e-2)) == len(vertex[:2]):
                these_sites.append([np.sign(vertex[0])*Lx*0.5, np.sign(vertex[1])*Ly*0.5, plane])
            ## check top & bottom boundaries
            elif np.isclose(abs(vertex[1]), np.array([0.5*Ly]), rtol=1e-3) :
                these_sites.append([vertex[0], np.sign(vertex[1])*Ly*0.5, plane])
            else : out_of_bounds += 1

    mobile_sites = pd.DataFrame(data=np.asarray(these_sites)).round(6).drop_duplicates().values

    if viz: print(f'# of points (mobile-ion sites) found: {len(mobile_sites)}. Outside: {out_of_bounds}')        
    
    ## remove duplicates around the edges of the box if needed, 
    ## this is using brute force
    to_remove = [];
    thresh = np.mean([Lx, Ly]) * 5e-3
    new_site_pts = [];
    for i, pt1 in enumerate(mobile_sites):
        if i in to_remove: continue
        for j, pt2 in enumerate(mobile_sites[i + 1:]):
            if Lx - abs(pt1[0] - pt2[0]) < thresh and abs(pt1[1] - pt2[1]) < thresh:
                # print pt1, pt2, i, j+i+1
                to_remove.append(j + i + 1)
            elif Ly - abs(pt1[1] - pt2[1]) < thresh and abs(pt1[0] - pt2[0]) < thresh:
                # print pt1, pt2, i, j+i+1
                to_remove.append(j + i + 1)
            elif Ly - abs(pt1[1] - pt2[1]) < thresh and Lx - abs(pt1[0] - pt2[0]) < thresh:
                to_remove.append(j + i + 1)
        new_site_pts.append(pt1)

    if viz: print(f'{len(mobile_sites)} points, removing at least {len(to_remove)} of them')
    mobile_sites = np.asarray(new_site_pts)
    
    ## if there are still too many mobile-ion sites
    if len(mobile_sites) > 2 * len(oxygens) :
        print('there are too many mobile-ion sites, most likely the corner is acting up')
        # r2 = mobile_sites[:,0] **2 + mobile_sites[:,1] **2
        # print(len(oxygens), sum(r2 > 0.25*Lx**2 + 0.25*Ly**2))

    return mobile_sites

# =============================================================================
# %% get the magnesium sites from atoms: find the conduction planes,
# ## then find the region between two conduction planes
# =============================================================================

def get_magnesium_sites(atoms, planes, Lz=1):
    '''
        returns the mg_sites, i.e. all Al positions to be replaced by Mg
        input atoms: data of atoms in .CONFIG file
        input planes: (approximate) z-coordinate of the conduction plane
        input Lz: overall length of the simulation box in z, assuming conduction 
                  planes are perpendicular to z
    '''
    # Convert planes to list
    planes = list(planes)
    planes.sort()

    # add the highest and lowest layer of conduction plane
    max_plane = max(planes) - Lz
    min_plane = min(planes) + Lz
    planes.append(max_plane)
    planes.append(min_plane)
    planes.sort()

    ## Find middle plane and round off to 2 d.p.
    middle_planes = []
    for i in range(0, len(planes) - 1):
        middle_plane = (planes[i] + planes[i + 1]) / 2
        middle_plane = round(middle_plane, 5)
        middle_planes.append(middle_plane)

    ## Calculate the maximum possible distance between Al 4 and Al 2a (or between Al 4 and 2b atoms)
    threshold = 0.02 * (3 / (len(planes) - 2)) * Lz  # may need to debug

    ## Find the Al 4, Al 2a and Al 2b positions based on query
    al_atoms = []

    ## For all Al atoms:
    for i in range(len(middle_planes)):
        plane = middle_planes[i]
        al_atom = atoms.query(f"atom == 'Al' & {plane} - {threshold} < z < {plane} + {threshold}")
        al_atoms.append(al_atom)

    ## Concatenate to merge the list back to a pd dataframe
    al_atoms = pd.concat(al_atoms)

    ## All possible planes for Al
    al_planes = al_atoms.z.unique()
    al_planes = al_planes.tolist()
    al_planes.sort()

    ## Separate the Al atoms into 4, 2a and 2b
    al4_atoms = []
    al2a_atoms = []
    al2b_atoms = []

    for i in range(len(al_planes)):
        current_plane = al_planes[i]
        if i == 0:
            al4_atom = al_atoms.query(f"z == {current_plane}")
            al4_atoms.append(al4_atom)
        elif i == 1:
            al2b_atom = al_atoms.query(f"z == {current_plane}")
            al2b_atoms.append(al2b_atom)
        elif (i - 2) % 3 == 0:
            al2a_atom = al_atoms.query(f"z == {current_plane}")
            al2a_atoms.append(al2a_atom)
        elif (i - 2) % 3 == 1:
            al4_atom = al_atoms.query(f"z == {current_plane}")
            al4_atoms.append(al4_atom)
        elif (i - 2) % 3 == 2:
            al2b_atom = al_atoms.query(f"z == {current_plane}")
            al2b_atoms.append(al2b_atom)

    ## Concatenate to merge the list back to a pd dataframe
    al4_atoms = pd.concat(al4_atoms)
    al2a_atoms = pd.concat(al2a_atoms)
    al2b_atoms = pd.concat(al2b_atoms)

    return al_atoms, al4_atoms, al2a_atoms, al2b_atoms, al_planes

# =============================================================================
# %% Get the number of Al atoms that will be substituted by Mg
# =============================================================================

def find_no_sub_al(atoms, x):
    '''
        returns an integer i.e. the no of Al atoms to be substituted
        input pd dataframe of atoms
        input x: the stoichiometry of Mg
    '''
    no_al_atoms = len(atoms.query('atom == "Al"').values)
    no_sub_al = no_al_atoms / 11 * x

    return no_sub_al

# =============================================================================
# %% A useful function to check if the file fulfills the criteria of
# the experiment
# =============================================================================

def file_compatibility(atoms, mg_to_be, planes, metal='Na'):
    '''
        returns the total number of mobile-ion atoms
        returns the number of Ag atoms per plane
        input pd dataframe of atoms
        input the number of "to be" Mg atoms
        input the number of conduction planes/the number of spinnel blocks
    '''
    ag_atoms = atoms.query('atom == @metal')
    ag_per_plane = len(ag_atoms) / len(planes)
    if ag_per_plane % 6 == 0:
        print(f"Yes, the # of {metal} atoms per plane is divisible by 6!")
    else:
        print(f".CONFIG file not suitable. Please use a file where the # of {metal} atoms per plane is divisible by 6!")

    if type(mg_to_be) == int or mg_to_be.is_integer():
        print("Yes, the no. of Mg atoms per plane to be substituted is divisible by the number of planes.")
        print("Yes, your file is compatible.")
    else:
        print(".CONFIG file not suitable. Please use a .CONFIG file with a suitable number of 'to be' Mg atoms, i.e."
              " the total is divisible by the number of planes!")

    return ag_atoms, ag_per_plane

# =============================================================================
# %% A useful function to get the [x y z] coordinates of a pandas data frame
# =============================================================================

def get_location(atoms):
    '''
        returns x y z coordinates of atoms
        input pd dataframe of atoms
    '''
    atoms = atoms[['x', 'y', 'z']].values
    return atoms

# =============================================================================
# %% A useful function to sort the [x y z] coordinates based on planes
#   (one plane per inner list.) It returns a list within a list.
# =============================================================================

def sort_planes(atoms, planes):
    '''
        returns a sorted list of [x y z] coordinates of atoms, listed according
        to plane
        input a list of of [x y z] coordinates of atoms
    '''

    result = []
    atoms = atoms.tolist()

    for i in range(len(planes)):
        temp_plane = []

        ## Sorting for each plane
        for j in range(len(atoms)):
            if atoms[j][2] == planes[i]:
                temp_plane.append(atoms[j])  # z-coordinate removed for convenience
        result.append(temp_plane)

    result = np.array(result)
    return result

# =============================================================================
# %% A useful function to get the [x y z] coordinates based on a single planes
#   (one plane per inner list.) It returns a single plane of atoms
# =============================================================================

def get_mg_site_per_plane(atoms, plane):
    '''
        returns a list of [x y z] coordinates of atoms, found at the z--coordinate of
        plane
        input a list of of [x y z] coordinates of atoms
        input the value of the plane (z-coordinate)
    '''

    result = []
    atoms = atoms.tolist()

    for layer in atoms:
        for atom in layer:
            if atom[2] == plane:
                result.append(atom)

    result = np.array(result)
    return result

# =============================================================================
# %% find the pairs of closest Al atoms to the given mid_oxygen site
# =============================================================================

def find_closest_als(atoms, mid_ox, plane, cell=np.eye(3), thresh=0.2):
    ''' return 2 closest Al ions to the given mid-oxygen sites
        input atoms: pandas database with atom names and x,y,z coordinates
        input mid_ox: x,y(,z) position, assumed fractional coordinates
        input plane: z position, assumed fractional coordinates
        input cell: 3x3 dimensions of the simulation box '''

    ## assuming an orthgonal cell, break down the cell to dimensions
    Lx = cell[0, 0];
    Ly = cell[1, 1];
    Lz = cell[2, 2]

    ## weighing factors for calculating distances
    wts = [10, 10, 1]  ## this over-emphasizes the distances in x,y
    t_xy = 0.01 * (Lx + Ly)  ## this is used in the pandas query
    thresh *= Lz

    ## add the average z if only x,y coordinates are given
    if len(mid_ox) < 3: mid_ox = [mid_ox[0], mid_ox[1], plane]

    als_1 = atoms.query(f'atom == "Al" & z > @plane - {thresh} & z < @plane + {thresh}')
    als_2 = als_1.query(f'x < @mid_ox[0] + {t_xy} & x > @mid_ox[0] - {t_xy}')
    als = als_2.query(f'y < @mid_ox[1] + {t_xy} & y > @mid_ox[1] - {t_xy}')

    ## calc distances from Al to the given O, weighing x & y more strongly 
    als['r'] = np.sqrt((als.x - mid_ox[0]) ** 2 * Lx ** 2 * wts[0] \
                       + (als.y - mid_ox[1]) ** 2 * Ly ** 2 * wts[1] \
                       + (als.z - mid_ox[2]) ** 2 * Lz ** 2 * wts[2])

    ## sort by closest
    als.sort_values(by=['r'], inplace=True)

    return als.iloc[:2]

# =============================================================================
# %% find nearest neighbors of mid-oxygen sites using edges
# ## This can probably be done in one line with networkx.ego_graph() ? 
# =============================================================================

def get_mid_oxygen_neighbors(fresh_mos, this_mo, past_mos, distance=1):
    ''' get distance-neighbors of a given mid-oxygen site recursively
        input fresh_mos: set of all mO points, each mO is [coordinates, edge],
                       each edge is a tuple (site index 1, site index 2) 
        input this_mo: one mO point 
        input past_nns: set of past nearest neighbors
        input distance: neighbors up to this distance (inclusive) to be returned

        returns are in same order as inputs: un-examined, then 1nn, then examined 
	'''

    #    print 'looking for {}-NN of'.format(distance), this_mo, 'past:', len(past_mos)

    ## safety case: this really should not happen
    if distance < 1:
        print('found a weird terminus at site:', this_mo)
        return fresh_mos, this_mo, past_mos

    ## real base case : get all the mO's that share elements of this_mo's edge tuple
    ## from the set of mO's that have not yet been found as neighbors (i.e. "fresh")
    if distance == 1:
        frontier_mos = set([x for x in fresh_mos if this_mo[0] in x or this_mo[1] in x])
        return fresh_mos - frontier_mos - {this_mo}, frontier_mos, past_mos.union({this_mo})

    ## recursion case
    elif distance > 1:
        ## first, get neighbors at a smaller distance: this is the recursion call
        fresh_in, last_shell, inside = get_mid_oxygen_neighbors(fresh_mos, this_mo, past_mos, distance - 1)

        frontier_mos = set()

        ## compile an extra set of all neighbors at the farthest distance
        for mo in last_shell:
            fresh_in, this_frontier, inside = get_mid_oxygen_neighbors(fresh_in, mo, inside, 1)
            frontier_mos = frontier_mos.union(this_frontier)

        ## return sets of sites: un-examined, then 1NN, then already-examined
        return fresh_in, frontier_mos, inside

# =============================================================================
# %% graph picking method using networkx - more useful for future-Mg-sites
#    than for mid-oxygen-sites: the former are nodes while the latter are edges
# =============================================================================

def pick_nodes_from_graph(edgelist, num_nodes, exclude=1, verbose=True, enforce_odd=False):
    '''
    pick_nodes_from_graph(edgelist, num_nodes, exclude=2, verbose=True)
    Pick nodes from a graph that is to be created from the input 'edgelist' to
    avoid dealing with copies or modifying any input graph in place. The number
    of nodes will be 'num_nodes', and each will be separated from the other 
    picked nodes by a graph radius of 'exclude' - assuming all edges are of 
    weight 1. Turn on the 'verbose' flag for print() statements as the method 
    goes along. FOR REPRODUCIBLE RESULTS, RESET SEED BEFORE RUNNING THIS!

    Parameters
    ----------
    edgelist : list of tuples
        This is the list of edges from which the graph will be constructed.
    num_nodes : int
        Number of nodes to pick / select.
    exclude : int, optional
        Distance by which the picked nodes are separated. The default is 1.
    verbose : boolean, optional
        Flag to print updates for every node picked. The default is True.
    enforce_odd : boolean, optional
        Flag to enforce an odd distance between consecutively picked nodes. 
        This helps to create a "symmetric" distribution in beta-doubleprime if needed.
        The default is False.

    Returns
    -------
    picked : set
        The nodes that are picked.
    fresh : set
        The nodes that are far enough from the picked nodes to be potentially 
        picked later.
    past : set
        The nodes that are within the 'exclude' radius of the already-picked 
        nodes and cannot be picked anymore.

    '''

    original_graph = nx.from_edgelist(edgelist)
    
    graph = nx.from_edgelist(edgelist)
    pick = None
    picked = set()
    past = set()

    for i in range(num_nodes):
        
        ## enforce odd graph distance between nodes if needed
        if i > 0 and enforce_odd:
            pickable_nodes = set([x for x in list(graph.nodes) if nx.shortest_path_length(original_graph,pick,x) % 2 != 0])
            pickable_nodes = list(set(graph.nodes).intersection(pickable_nodes))
        else:
            pickable_nodes = list(graph.nodes)
            
        ## Pick a random node
        pick = random.choice(pickable_nodes)
        if verbose: print(f'picking node {pick}')
        
        ## add the node to picked set
        picked = picked.union({pick})

        ## make a graph of specified radius: this will be excluded
        this_past = nx.ego_graph(graph, pick, radius=exclude)

        ## remove nodes from the fresh graph
        graph.remove_nodes_from(list(this_past.nodes))

        ## add ego_graph to excluded
        past = past.union(set(this_past.nodes))
        past -= {pick}

        ## handle emptiness
        if len(graph.nodes) < 1:
            print(f'ran out of nodes, picked {i + 1} of {num_nodes}, returning.')
            break

        ## more mundane debugging messages
        if verbose: print(f'picked: {i + 1}, excluded: {len(past)}, remaining: {len(graph.nodes)}')

    ## return unpicked nodes
    fresh = set(graph.nodes)

    return picked, fresh, past

# =============================================================================
# %% Calculate path lengths in a graph
# =============================================================================

def path_lengths(graph):
    
    num_nodes = len(graph.nodes)
    
    paths = np.zeros((num_nodes, num_nodes))
    
    path_dict = dict(nx.all_pairs_shortest_path_length(graph))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            paths[i,j] = path_dict[i][j]
    
    return paths

# =============================================================================
# %% generate random permutations of mid-ox sites based on excluding neighbors
#    (Deprecated in favor of using the freud AABBQuery API)
# =============================================================================

def generate_mid_oxygens(mid_oxs, num_sites, dist_exclude=3, verbose=False):
    ''' starting with a random origin index, keep generating random mid-oxygen sites 
        until reaching the num_sites target or running out of sites, and 
        excluding a network-radius of dist_exclude.
        input mid_oxs: the starting list of all mO points, each mO is [coords, edge],
                       each edge is tuple (site index 1, site index 2)
        input origin_idx: integer in range(len(mid_oxs))
        input num_sites : how many to generate
        input dist_exclude: how much distance each neighbor eats up
        RESET SEED BEFORE CALLING THIS!
	'''

    ## don't bother with the real-space coordinates, just take the graph edges
    all_sites = set([x[1] for x in mid_oxs])

    ## at first, all sites are possible to pick, so they are in the 'fresh' set
    fresh_to_pick = set([x[1] for x in mid_oxs])

    ## the sets for 'picked' and 'past' i.e. discarded sites
    picked = set()
    all_past = set()

    ## pick sites
    for i in range(num_sites):
        ## if there are sites to pick from, then pick one
        if fresh_to_pick:
            pick = random.choice(tuple(fresh_to_pick))

            ## no replacement: remove the picked site from 'fresh'
            fresh_to_pick -= {pick}

            ## add the picked site to the output
            picked = picked.union({pick})
            if verbose: print('adding:', pick)

            ## find all neighbors of the newly-picked site within exclusion radius
            _, frontier, past = get_mid_oxygen_neighbors(all_sites, pick, set(), dist_exclude)

            ## add the excluded sites to 'past'/discarded, remove from 'fresh'
            all_past = all_past.union(past.union(frontier))
            fresh_to_pick -= all_past
            if verbose: print(f'now excluding {len(all_past)} total sites')
        ## if ran out of sites and can pick no more, say so
        else:
            print(f'out of sites to try, picked {len(picked)}')
            break

    ## return stuff
    return picked, fresh_to_pick, all_past

# =============================================================================
# %% generate quasi-random permutations of mid-ox sites based on excluding neighbors
#    and also trying to pack close by favoring placements at the min distance
# =============================================================================

def generate_mid_oxygens_packed(mid_oxs, num_sites, dist_exclude=3, verbose=False):
    ''' starting with a random origin index, keep generating random mid-oxygen sites 
        until reaching the num_sites target or running out of sites, and 
        excluding a network-radius of dist_exclude.
        input mid_oxs: the starting list of all mO points, each mO is [coords, edge],
                       each edge is tuple (site index 1, site index 2)
        input origin_idx: integer in range(len(mid_oxs))
        input num_sites : how many to generate
        input dist_exclude: how much distance each neighbor eats up
        RESET SEED BEFORE CALLING THIS!
	'''
    
    ## don't bother with the real-space coordinates, just take the graph edges
    all_sites = set([x[1] for x in mid_oxs])

    ## at first, all sites are possible to pick, so they are in the 'fresh' set
    fresh_to_pick = set([x[1] for x in mid_oxs])

    ## the sets for 'picked' and 'past' i.e. discarded sites
    picked = set()
    all_past = set()
    nearest_fresh = [] ## going to be a list of sets

    ## pick sites
    for i in range(num_sites):
        ## if there are sites to pick from, then pick one
        if fresh_to_pick:
            if not nearest_fresh : pick = random.choice(tuple(fresh_to_pick))
            else: 
                pick = random.choice(tuple(nearest_fresh[-1]))
                for s in range(len(nearest_fresh)) : nearest_fresh[s] -= {pick}

            ## no replacement: remove the picked site from 'fresh'
            fresh_to_pick -= {pick}

            ## add the picked site to the output
            picked = picked.union({pick})
            if verbose: print('adding:', pick)

            ## find all neighbors of the newly-picked site within exclusion radius
            _, frontier, past = get_mid_oxygen_neighbors(all_sites, pick, set(), dist_exclude)

            ## get closest-possible edges that can be picked using frontier
            this_nearest_fresh = set().union(*[get_mid_oxygen_neighbors(fresh_to_pick, f, past)[1] for f in frontier]) - past
            # print(this_nearest_fresh)
            
            ## update nearest_fresh list of sets
            if not nearest_fresh : nearest_fresh.append(this_nearest_fresh)
            else: 
                carryover = this_nearest_fresh
                for s in range(len(nearest_fresh)) :
                    nearest_fresh[s] = list(nearest_fresh[s]) + list(carryover)
                    carryover = set([k for k,v in Counter(nearest_fresh[s]).items() if v > 1])
                    nearest_fresh[s] = set(nearest_fresh[s])
                if carryover : nearest_fresh.append(carryover)
            # print(nearest_fresh)
            
            ## add the excluded sites to 'past'/discarded, remove from 'fresh'
            all_past = all_past.union(past.union(frontier))
            fresh_to_pick -= all_past
            for s in range(len(nearest_fresh)) : 
                nearest_fresh[s] -= all_past
            nearest_fresh = [s for s in nearest_fresh if s]
            if verbose: 
                print(f'now excluding {len(all_past)} total sites, {len(nearest_fresh[-1]) if nearest_fresh else 0} on deck')
                # print(nearest_fresh)
        ## if ran out of sites and can pick no more, say so
        else:
            print(f'out of sites to try, picked {len(picked)}')
            break

    ## return stuff
    return picked, fresh_to_pick, all_past

# =============================================================================
# %% find empty mobile-ion sites for placing ions in them
# ## this is probably better computable with freud voronoi and nearest neighbors,
# ## as for hop tracking
# =============================================================================

def find_empty_brabr(atoms, plane, cell=np.eye(3)):
    ''' 
    returns list of (x,y) coordinates of empty mobile-ion sites
        
    input atoms: pandas database with atom names and x,y,z coordinates
    input plane: z position, assumed fractional coordinates
    '''

    ## assuming an orthgonal cell, break down the cell to dimensions
    Lx = cell[0, 0];
    Ly = cell[1, 1];
    Lz = cell[2, 2]

    thresh = 0.01 * Lz  ## crude for in-plane
    sites = get_mobile_ion_sites(atoms, plane, cell)

    empties = []

    mobile_ions = atoms.query('atom != "O" & atom != "Al" & z > @plane - @thresh & z < @plane + @thresh')

    print(f'found {len(mobile_ions)} mobile ions in plane {plane:.3f}')
    
    ## check which sites are empty with a bounded-box query
    box = freud.box.Box.from_matrix(cell)
    que = freud.locality.AABBQuery(box, mobile_ions[['x','y','z']].values)
    query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=False)
    result = que.query(sites, query_args)
    
    ## keep only the sites with large distances
    for s, (i,j,distance) in zip(sites, result):
        # if (-0.50 * Lx <= s[0] < 0.50 * Lx) and (-0.50 * Ly <= s[1] < 0.50 * Ly):
        #     mobile_ions.loc[:, 'r'] = np.sqrt((mobile_ions.x - s[0]) ** 2 + (mobile_ions.y - s[1]) ** 2)
        #     if mobile_ions.r.min() > thresh: empties.append(s)
        if distance > Lx/200 and distance > Ly/200 : empties.append(s)
            
    print(f'empty sites: {len(empties)}')

    return np.asarray(empties), mobile_ions

# =============================================================================
# %% substitute picked aluminium sites with magnesium
# =============================================================================
def substitute_aluminium(atoms, picked):
    ''' substitutes the picked aluminium sites with magnesium,
        moves the aluminium ions for the Roth defect, add 2 mobile ions.
        input atoms: pandas dataframe with index idx, atom names, and x,y,z
        input picked: positions x, y, z for the picked aluminium sites '''
    indices = []
    for i in range(len(picked)):
        x_coord = picked[i, 0]
        y_coord = picked[i, 1]
        z_coord = picked[i, 2]
        idx = atoms.query(f'x=={x_coord} & y=={y_coord} & z=={z_coord}').index.tolist()
        idx = idx[0]
        atoms.loc[idx, 'atom'] = 'Mg'
        indices.append(idx)
    return atoms, indices

# =============================================================================
# %% add a Roth defect: add one interstitial oxygen, and move 2 Al's
# =============================================================================

def create_roth_defects(atoms, plane, mo_sites, cell, thresh=0.4, verbose=False):
    ''' adds an interstitial oxygen at the specified mid-oxygen position,
        moves the aluminum ions for the Roth defect, adds 2 mobile ions.
        input atoms: pandas dataframe with index idx, atom names, and x,y,z 
        input plane: z-coordinate (approximate) 
        input mo_sites: positions x,y for the interstitials (not site edge)
        input verbose: boolean flag to print steps for debugging
        returns: appended atoms, and a list of indices for newly added interstitial oxygens '''

    max_idx = atoms.index.values.max()
    mobile = list(set(atoms.atom.unique()) - {'O', 'Al', 'Mg'})[0]
    thresh *= cell[2, 2]
    num_defects = len(mo_sites)
    
    ## find empty sites - done every time to avoid overlaps between new atoms
    empties, _ = find_empty_brabr(atoms, plane, cell)
    empties = pd.DataFrame({'x': empties[:, 0], 'y': empties[:, 1], 'z': empties[:, 2]})
    if len(empties) < 2. * len(mo_sites): print('not enough empty sites')
    
    # new_atoms = pd.DataFrame()
    rows_list = []
    
    ## make a row for oxygen
    for i, mo in enumerate(mo_sites):
        
        ## make an oxygen row
        rows_list.append({'idx': max_idx + 1 + i, 'atom': 'O', 'x': mo[0], 'y': mo[1], 'z': plane})  ## oxygen

        if verbose: print(f'     mO:     ({mo[0]:.3f}, {mo[1]:.3f}, {plane:.3f})')

        ## move aluminum ions
        for al in find_closest_als(atoms, mo, plane, cell, thresh).iterrows():
            if verbose:
                print('Al {:4d}: old ({:.3f}, {:.3f}, {:.3f}), new z = {:.3f}'.format(
                    al[0], al[1].x, al[1].y, al[1].z,(al[1].z + plane) * 0.5))

            ## actually move; this works out for stoichiometric beta input file
            atoms.loc[al[0], 'z'] = (al[1].z + plane) * 0.5
    
        # ## add 2 mobile ions to empty mobile ion sites closest to the Oi
        # empties['r'] = np.sqrt((empties.x - mo[0]) ** 2 + (empties.y - mo[1]) ** 2)
        # closest = empties.sort_values(by=['r']).iloc[1:3] # .reset_index()
        # for ii, s in enumerate(closest.itertuples()):
        #     rows_list.append({'idx': max_idx + i * 3 + 2 + ii, 'atom': mobile, 'x': s.x, 'y': s.y, 'z': s.z})
        # empties = empties.append(closest, ignore_index=True).drop_duplicates(subset=['x','y'],keep=False)

    ## place new mobile ions randomly onto empty sites
    for i,s in zip(range(num_defects+1,num_defects*3+1), random.sample(range(len(empties)), num_defects*2)) :
        site = empties.iloc[s]
        rows_list.append({'idx': max_idx + i , 'atom': mobile, 
                          'x': site.x, 'y': site.y, 'z': site.z, 'id':max_idx+i})

    new_atoms = pd.DataFrame(rows_list).set_index('idx').sort_index()
    print(f'adding {len(new_atoms)} atoms with indices {new_atoms.index.values.min()}-{new_atoms.index.values.max()}')

    return atoms.append(new_atoms), new_atoms.query('atom == "O"').index.values.tolist()
    # return atoms, new_atoms.query('atom == "O"').index.values.tolist()

# =============================================================================
# %% generate a grid of points from a beta double prime structure
# ## Modified from 2019 code that uses 'grid' as the vertices of the mobile ion
# ## site polygons.
# ## The atoms dataframe should be generated from the structure file that is
# ## directly fed into the simulation. Otherwise the conversions between CIF,
# ## CONFIG, and LAMMPS may change the indices of atoms and sites.
# =============================================================================

def generate_grid(atoms, plane, cell=np.eye(3), viz=True, fractional=True):
    ''' generate a simple CSV-able list of x,y coordinates of mobile-ion grid points
        These are the vertices of mobile ion site polygons, shifted to fit into
        a box that is 0 to L. The output is 0 to L.
        input atoms: pandas dataframe with atom identities and positions within (-L/2, +L/2)
        input plane: z-coordinate of a conduction plane
        input cell: leave as np.eye(3) if atoms already have fractional coords
        '''

    box = freud.box.Box(Lx=cell[0, 0], Ly=cell[1, 1], is2D=True)  ## L=1
    
    if viz:        
        ## this works - but for some reason freud does not want to compute it
        ## so it is here only for visualization
        _ = get_mobile_ion_sites(atoms, plane, cell, viz=viz)

    ## get the oxygens in the conduction plane
    thresh = 0.04
    oxygens = atoms.query(f'atom == "O" & z > {plane - thresh} & z < {plane + thresh}')
    
    # check for interstitial oxygens in the offhand case they did not get
    # "ignored" when importing a lammps file
    if 'type' in oxygens.columns :
        if len(oxygens.type.unique()) > 1:
            oi_type = oxygens.type.astype(int).max()
            oxygens = oxygens.query(f'type < {oi_type}')
    
    oxygen_sites = oxygens[['x', 'y', 'z']].values
    oxygen_sites[:,-1] = 0

    ## extend the points using periodic buffer (freud 2.2.0)
    pbuff = freud.locality.PeriodicBuffer()
    # pbuff.compute(system=(box, these_sites), buffer=0.15 * (box.Lx + box.Ly), images=False)
    pbuff.compute(system=(box, oxygen_sites), buffer=0.05 * (box.Lx + box.Ly), images=False)

    ## concatenate the original points and the extended points
    gridpts = np.concatenate((oxygen_sites, pbuff.buffer_points), axis=0)
    
    ## plot new grid points
    if viz: plt.gca().scatter(gridpts[:,0], gridpts[:,1], s=75, c='k', label='Oxygen')

    ## shift from freud coordinate convention (-0.5 to 0.5) to LAMMPS (0 to 1)
    xs = gridpts[:, 0] + 0.5 * box.Lx
    ys = gridpts[:, 1] + 0.5 * box.Ly

    ## convert to fractional if needed
    if fractional and max(xs) > 2:
        xs /= cell[0, 0]
        ys /= cell[1, 1]
        if cell.max() == 1.:
            print('=====\ngenerate_grid() warning: pass a cell to divide real coords\n=====')

    ## the rounding is necessary; there may be duplicate points otherwise
    rounding = [3, 6][fractional]
    return pd.DataFrame(data={'x': xs, 'y': ys}).round(rounding).drop_duplicates().sort_values(by=['x','y'])

# =============================================================================
# %% functions retrieving pre-saved network / graph paths from files
# =============================================================================

# def site_paths(folder='.', zs=['z_all', '012', '037', '062', '087']):
#     ''' retrieves number_of_polygons x number_of_polygons arrays with network path lengths
#         between mobile-ion sites from pre-saved files. All this is only in 
#         reference to a pre-defined "grid" that defines the network. 
#         AP moved from hop_utils 2020/06/23 '''
#     paths = dict()

#     for zz in zs:
#         try:
#             paths[zz] = np.loadtxt(folder + '/paths{}.csv'.format(zz), delimiter=',').astype(int)
#         except:
#             print('missing paths at z = {}'.format(zz))

#     return paths


# def o_sites(folder='.', zs=['z_all', '012', '037', '062', '087']):
#     ''' retrieves from pre-saved files the 1D arrays with network indices sites
#         that have an edge that is occupied by an O_interstitial. All this is only
#         in reference to a pre-defined "grid" that defines the network.
#         AP moved from hop_utils 2020/06/23 '''
#     paths = dict()

#     for zz in zs:
#         try:
#             paths[zz] = np.loadtxt(folder + '/oxygen_cells_{}.csv'.format(zz), delimiter=',').astype(int)
#         except:
#             print('missing O sites z = {}'.format(zz))

#     return paths


# def BR_sites(folder='.', zs=['z_all', '012', '037', '062', '087']):
#     ''' retrieves (from pre-saved files) the 1D arrays with network indices of 
#         Beevers-Ross sites for beta-aluminas. All this is only
#         in reference to a pre-defined "grid" that defines the network.
#         AP moved from hop_utils 2020/06/23 '''
#     BR_dict = dict()

#     for zz in zs:
#         try:
#             BR_dict[zz] = np.loadtxt(folder + '/sites BR {}.csv'.format(zz), delimiter=',')
#         except:
#             print('missing BR sites at z = {} in folder {}'.format(zz, folder))

#     return BR_dict

# =============================================================================
# %% visualization function from freud. This method started here:
# ## freud.readthedocs.io/en/v1.2.0/examples/module_intros/Voronoi-Voronoi.html
# ## AP copied from hop_utils to crystal_utils on 2020/06/23
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
# %% make freud Voronoi & NearestNeighbors objects from a grid of points (x,y)
# ## this method assumes that the box is square
# =============================================================================

def make_voronoi(grid_path, L=1., z=0, debug=False):
    '''
    input: grid_path is the (relative) path to the file with grid points
    This works in fractional coordinates, and assumes grid is in fractional coordinates
    '''

    pts = pd.read_csv(grid_path)
    gridpts = np.array([pts.x, pts.y]).T
    print(f'read {len(gridpts)} points')

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

    print(f'{len(site_pts)} points, removing {len(to_remove)} of them')

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
# %% General method for finding neighboring sites in a 3D box w/ freud
# ## Should be usable for both mobile-ion sites and for Mg/Al sites
# =============================================================================

def get_neighboring_points(points, cell=np.eye(3), num_nn=3):
    ''' 
    Finds several ('num_nn') nearest neighbors for every point in 'points' 
    within a 'cell' (fractional or real-dimensioned, with all boundaries 
    assumed periodic)

    Parameters
    ----------
    points : numpy array pf shape (n,3)
        Coordinates of points whose neighbor-hood is of interest
    cell : numpy array of shape (2,2) or (3,3), optional
        Shape of the simulation cell and freud box. The default is np.eye(3).
    num_nn : int, optional
        Number of neighbors to return for each point
        
    Returns
    -------
    edges : list of tuples
        every tuple is the indices of neighboring points.
    distances : list of floats
        every distance corresponding to a nearest-neighbor edge
    '''

    ## TODO: figure out whether coordinates of the input sites are fractional 
    ## or real-spaced. For now, assume fractional with the default np.eye() cell 

    ## construct the freud box
    box = freud.box.Box.from_matrix(cell)
    ## Wrap the given points just in case - but for now assuming that all 
    ## points are coming from inside the box by default, such as atoms being
    ## inside their simulation box. Removed to see if ordering gets changed.
    # points = box.wrap(points)

    ## compose query parameters
    query_args = dict(mode='nearest', num_neighbors=num_nn, exclude_ii=True)
    
    ## create the freud query and query it (freud 2.2.0)
    que = freud.locality.AABBQuery(box, points)
    result = que.query(points, query_args)

    ## remove duplicate neighbors from the result - also returnable by 
    ## making that into a 2D np.array and then np.argwhere() 
    edges = list()
    distances = list()

    for row in list(result):
        if (row[1], row[0]) not in edges:
            edges.append((row[0], row[1]))
            distances.append(row[2])
    ## remove result from return once done
    return edges, distances

# =============================================================================
# %% General method for finding nearest (x, y) sites in a 3D box w/ freud
# ## between the mobile-sites and picked Mg_sites
# =============================================================================

def get_nearest_points(site_pts, defect_pts, cell=np.eye(3), num_nn=6):
    '''
    Finds several ('num_nn') nearest neighbors within the iterable 'site_points' 
    for every point in the iterable 'defect_points'. This is wrapped inside
    a 'cell' (fractional or real-dimensioned, all boundaries assumed periodic)
    
    Two kinds of neighbors are returned: 
        '0' directly above/below, 
        '1' offset in xy

    Parameters
    ----------
    site_points : numpy array pf shape (n,3)
        Coordinates of points that are the "neighbors" being found
    defect_points : numpy array pf shape (n,3)
        Coordinates of points whose neighbors are being found
    cell : numpy array of shape (2,2) or (3,3), optional
        Shape of the simulation cell and freud box. The default is np.eye(3).
    num_nn : int, optional
        Number of neighbors to return for each point

    Returns
    -------
    edges : list of tuples
        every tuple is the indices of neighboring points.
    distances : list of floats
        every distance corresponding to a nearest-neighbor edge
    '''

    ## TODO: figure out whether coordinates of the input sites are fractional
    ## or real-spaced. For now, assume fractional if the default np.eye() cell is passed

    ## construct the freud box
    box = freud.box.Box.from_matrix(cell)
    
    ## Wrap the given points just in case - but for now assuming that all
    ## points are coming from inside the box by default, such as atoms being
    ## inside their simulation box. Removed to see if ordering gets changed.
    ## This has not been needed.
    # points = box.wrap(points)

    ## compose query parameters
    query_args = dict(mode='nearest', num_neighbors=num_nn, exclude_ii=False)

    ## create the freud query and query it (freud >= 2.2.0)
    que = freud.locality.AABBQuery(box, site_pts)  # site_pts = mobile ion sites
    result = que.query(defect_pts, query_args)  # defect_pts = picked Mg atoms

    ## remove duplicate neighbors from the result - also returnable by
    ## making that into a 2D np.array and then np.argwhere()
    ## NOTE: if using this method, need to account for box boundaries
    edges_0 = list()
    distances_0 = list()
    edges_1 = list()
    distances_1 = list()
    # blacklist_mg = []
    # for row in list(result):
    #     # if (row[1], row[0]) not in edges:
    #     ## compute the (x,y) distances and convert to fractional coordinates
    #     horizontal_distance_x = abs(defect_pts[row[0], 0] - site_pts[row[1], 0]) 
    #     horizontal_distance_y = abs(defect_pts[row[0], 1] - site_pts[row[1], 1]) 
    #     horizontal_distance = np.hypot(horizontal_distance_x, horizontal_distance_y)
    #     ## nearest_points is a tuple: (query_point_index, point_index)
    #     if horizontal_distance < 0.01 * np.mean([cell[0,0], cell[1,1]]):  # For Al(4) and Al(2a) atoms
    #         edges_0.append((row[0], row[1]))
    #         distances_0.append(row[2])
    #         blacklist_mg.append(row[0])

    # ## Note the arbitrary cutoff at 6.5 for non-fractional imports
    # for row in list(result):  # For Al(2b) atoms
    #     horizontal_distance_x = abs(defect_pts[row[0], 0] - site_pts[row[1], 0]) 
    #     horizontal_distance_y = abs(defect_pts[row[0], 1] - site_pts[row[1], 1])
    #     horizontal_distance = np.hypot(horizontal_distance_x, horizontal_distance_y)
    #     # i.e. if the found Ag atom does not have a neighboring Mg that is an Al(4) or Al(2a) site
    #     # i.e. if it's an Al 2b site
    #     if row[0] not in blacklist_mg and 0 < horizontal_distance and row[2] < 6.4:
    #         # if (row[1], row[0]) not in edges_1:
    #         edges_1.append((row[0], row[1]))
    #         distances_1.append(row[2])
                
    ## classify the neighbors of each defect
    ## if all distances are within a small value of each other (0.007 * cell[2,2])
    for defect in range(len(defect_pts)):
        rows = [row for row in result if row[0] == defect]
        distances = [row[2] for row in rows]
        # if min(distances) < 10 : print(defect, np.unique(np.round(distances,4)))
        if max(distances) - min(distances) < 0.007 * cell[2,2] : ## should work for real+frac
            # if min(distances) < 10 : print(defect, np.unique(np.round(distances,4)), 1) ## distance 1, take all neighbors
            for row in rows:
                edges_1.append((row[0], row[1]))
                distances_1.append(row[2])
        else : 
            # if min(distances) < 10 : print(defect, np.unique(np.round(distances,4)), 0) ## distance 0, take min distance
            for row in rows :
                if row[2] == min(distances):
                    edges_0.append((row[0], row[1]))
                    distances_0.append(row[2])
            

    ## DEBUGGING STATEMENTS:
    # print("Long Al(4) and Al(2a) edges are", [x for (x,d) in zip(edges,distances) if d > 6.5])
    # print("Long Al(2b) edges are", [x for (x,d) in zip(edges_1,distances_1) if d > 6.])

    # edges3 = edges + edges2
    # distances3 = distances + distances2

    ## remove result from return once done
    return edges_0, edges_1, distances_0, distances_1

# =============================================================================
# %% Method for getting the index of atoms where edges lie in, using
# ## pandas dataframe only (USE THIS IF YOU HAVE NOT SPLICED ANY LISTS)
# =============================================================================

def get_index_edges(edges, atoms):
    '''
        returns tuples containing the index of two atoms where the edges lie
        inputs a list of edges (tuples with the ordinal number of atoms)
    '''
    result = []
    index = atoms.index.tolist()
    for edge in edges:
        index_tuple = (index[edge[0]], index[edge[1]])
        result.append(index_tuple)
    return result

# =============================================================================
# %% Method for getting the original index of atoms where edges lie in, using
# ## BOTH spliced lists and pandas dataframe
# ## this is not needed
# =============================================================================

# def get_index_edges_2(edges, query_atoms, target_atoms, atoms):
#     '''
#         returns tuples containing the index of two atoms where the edges lie
#         inputs edges: a list of edges (tuples with the ordinal number of atoms)
#         inputs query_atoms: the list of values from which the first value of a
#                             tuple in "edges" are spliced from
#         inputs target_atoms: the list of values from which the second value of a
#                             tuple in "edges" are spliced from
#         inputs atoms: the pd dataframe of all the atoms
#     '''
#     result = []
#     for i in range(len(edges)):
#         ## Find the x y z values of a given query atom based on ordinal values
#         at_q = query_atoms[edges[i][0]]
        
#         ## query for the index based on the x y z values of a given query atom
#         index1 = atoms.query(f'x=={at_q[0]} and y=={at_q[1]} and z=={at_q[2]}').index.tolist()
#         index1 = index1[0]

#         ## Find the values of a given target atom based on ordinal values
#         at_t = target_atoms[edges[i][1]]

#         ## query for the index based on the x y z values of a given target atom
#         index2 = atoms.query(f'x=={at_t[0]} and y=={at_t[1]} and z=={at_t[2]}').index.tolist()
#         index2 = index2[0]
#         result.append((index1, index2))

#     return result

# =============================================================================
# %% method to calculate mid-oxygen sites using new freud (2.2.0) tools
# =============================================================================

def get_mid_oxygen_sites_freud(mobile_ion_sites, cell=np.eye(3), viz=True):
    '''
    Generates mid-oxygen sites using freud

    Parameters
    ----------
    mobile_ion_sites : numpy array of shape (n,3)
        Coordinates of points at centers of mobile-ion sites (3D or 2D w/ z=0).
    cell : numpy array of shape (2,2) or (3,3), optional
        Shape of the simulation cell and freud box. The default is np.eye(3).
    viz : boolean, optional
        Flag whether to plot sites and midpoints. The default is True.

    Returns
    -------
    mid_oxs : list of tuples
        midpoints between neighboring input sites: coordinates + edge with 
        indices of the input sites of which it is a midpoint.
    edges : list of tuples
        every tuple is the indices of neighboring mobile-ion sites. 
    midpts : list of tuples
        every tuple is coordinates of the midpoint.

    '''

    ## initialize lists for sites and midpoints
    sites = list()

    ## for the assumed orthogonal box, simplify dimensions to check that the
    ## input sites are inside it
    Lx = np.round(cell[0, 0],4)
    Ly = np.round(cell[1, 1],4)
    
    ## if a 3D cell was passed
    if len(cell) > 2: Lz = np.round(cell[2, 2],4) 
    else: Lz = 1
        
    ## if a 3D box is passed, but the sites are 2D: append zeros in z
    if mobile_ion_sites.shape[1] < 3:
        temp = np.zeros((len(mobile_ion_sites),3))
        temp[:,:-1] = mobile_ion_sites
        mobile_ion_sites = temp
            
    ## Append extra zeros in z if needed, and filter out any extras that may 
    ## have come from outside the simulation box from the wrapping of an earlier 
    ## freud Voronoi object. 
    ## 2020/07/25: switched to in_bounds(), which adds <= at upper boundaries
    for s in mobile_ion_sites:
        if in_bounds(s, np.array([Lx,Ly,Lz])*-0.5, np.array([Lx,Ly,Lz])*0.5):
            sites.append(s)
            
    sites = np.asarray(sites)
    if viz: print(f'# pts (mobile-ion sites) inside ortho cell: {len(sites)}')
    # sites = np.asarray(sites)
    
    # print(sites[:10])
    # print(type(sites))
    # print(sites.shape)

    ## create a new freud Voronoi object
    box = freud.box.Box(Lx=cell[0, 0], Ly=cell[1, 1], is2D=True)
    sites = box.wrap(sites) # removed 2020/07/25, somehow 
    
    site_pts = np.zeros((len(sites), 3))  ## should not be necessary
    site_pts[:, :2] = sites[:, :2]

    site_vor = freud.locality.Voronoi(box)
    site_vor.compute(system=(box, site_pts))

    ## PLot the voronoi first if plotting at all
    if viz:
        fig, ax = plt.subplots(figsize=(8, 7))
        draw_voronoi(box, site_pts, site_vor.polytopes, # draw_points=True,
                     cell_numbers=range(len(site_pts)), draw_box=True)

    ## get nearest neighbors using freud
    ## cutoff (3) on the number of nearest neighbors accommodates inexactness,
    ## such as if computing from a snapshot of an MD trajectory
    edges, _ = get_neighboring_points(sites, cell, num_nn=3)

    ## try getting midpoints via wrapped differences in distance between nn's
    nn_vectors = np.asarray([sites[edge[1]] - sites[edge[0]] for edge in edges])
    nn_vectors = box.wrap(nn_vectors)
    midpts = [sites[edges[i][0]] + nn_vectors[i] * 0.5 for i in range(len(edges))]
    midpts = box.wrap(np.asarray(midpts))

    ## Plot the midpoints between sites
    if viz:
        mid_sites = np.asarray(midpts)
        print(f'# midpoints between mobile-ion sites: {len(mid_sites)}')
        # ax.scatter(mid_sites[:, 0], mid_sites[:, 1], c='tab:orange')

    mid_oxs = [x for x in zip(midpts, edges)]

    return mid_oxs, edges, midpts

# =============================================================================
# %% automatically detect BR sites vs aBR sites vs beta-doubleprime sites
# ## There are 5 types of nearest-neighbor environments for mobile-ion sites:
# ## - BR : 3 above and 3 below, distance ~2.88 AA
# ## - aBR: 1 above and 1 below, distance ~2.38 AA
# ## - BR  +Oi : an extra close neighbor at 1.61 AA
# ## - aBR +Oi : an extra close neighbor at 1.61 AA
# ## - beta": 1 above and 3 below, or vice versa, distances 2.38 / 2.65 AA
# =============================================================================

def auto_get_BR_sites(atoms, cell, site_pts, atoms_are_frac=False):
    '''
    Automatically identify the types of sites at the coordinates of 'site_pts' 
    using the numbers of oxygens at distances between 2 and 3 angstroms.
    
    This has not been tested for fractional-coordinate inputs, but should work.
    Obviously, the 'atoms' coordinates and the 'site_pts' coordinates should be
    the same: either both real-space, or both fractional.

    Parameters
    ----------
    atoms : pandas DataFrame
        DataFrame with atoms. Only 'O' is queried, so in principle the others 
        do not need to be passed.
    cell : numpy array, shape (3,3)
        Dimensions of the simulation box in real-space coordinates.
    site_pts : numpy array, shape (n,3)
        Coordinates of one or more sites.
    atoms_are_frac : boolean, optional
        Flag for whether the coordinates are fractional. The default is False.

    Returns
    -------
    site_types : list
        List of types of sites ("BR", "aBR", "doubleprime", or "error") in 
        order of their appearance in 'site_pts'.

    '''
    
    ## dictionary with keys = numbers of nearest oxygens, values = sites
    sites = {2:'aBR', 4:'doubleprime', 6:'BR'}
    
    ## convert to real-space distances
    if atoms_are_frac:
        atoms.x *= cell[0,0]
        atoms.y *= cell[1,1]
        atoms.z *= cell[2,2]
        
        site_pts[:,0] *= cell[0,0]
        site_pts[:,1] *= cell[1,1]
        site_pts[:,2] *= cell[2,2]
    
    ## create box from cell
    box = freud.box.Box.from_matrix(cell)
    
    ## compose query parameters
    query_args = dict(mode='ball', r_max = 3., r_min = 2., exclude_ii=False)
    
    ## create the freud query (freud 2.2.0)
    que = freud.locality.AABBQuery(box, atoms.query('atom == "O"')[['x','y','z']].values)
    
    ## output
    site_types = list()
    
    ## if only one point is input
    if len(site_pts.shape) < 2:
        result = len(list(que.query(site_pts, query_args)))
        return sites[result] if result in sites.keys() else 'error'
    
    ## query for each point - hope this works. Might not be the fastest
    for pt in site_pts:
        result = len(list(que.query(pt, query_args)))
        site_types.append(sites[result] if result in sites.keys() else f'error {result}')
        # if result > max(sites.keys()) : print(result)
        # if result < min(sites.keys()) : print(result)
        
    return site_types
        
# =============================================================================
# %% get mobile ion sites adjacent to oxygen interstitials
# =============================================================================

def get_defect_adjacent_sites(cell, site_pts, defects, query_args=None):
    '''
    Return indices of sites that are closest to the defects. This is a general
    wrapper on the freud AABBQuery method, and will accept arbitrary 'query_args'.

    Parameters
    ----------
    cell : numpy array, shape (n,3)
        Dimensions of the simulation box.
    site_pts : numpy array, shape (n,3)
        Coordinates of the sites that may or may not be closest to defects.
    defects : numpy array, shape (n,3)
        Coordinates of the defects.
    query_args : dict, optional
        Arguments to pass to the freud query. The default is None.

    Returns
    -------
    adjacent_sites : numpy array, shape (n,)
        Integer ordinal indices of the sites adjacent to defects.
    distances : numpy array, shape (n,)
        Distances between the adjacent sites and their nearest defect.

    '''

    ## construct the freud box
    box = freud.box.Box.from_matrix(cell)

    ## Compose query parameters. Default is set up for beta: 2NN and nearest.
    if query_args is None:
        query_args = dict(mode='nearest', num_neighbors=2, exclude_ii=False)
    
    # create the freud query (freud 2.2.0)
    que = freud.locality.AABBQuery(box, site_pts)
    
    ## query and return indices
    adjacent_result = que.query(defects, query_args=query_args)
    adjacent_sites = np.array(list(adjacent_result))
    distances = adjacent_sites[:,-1]
    adjacent_sites = adjacent_sites[:,1].astype(int)
    
    return adjacent_sites, distances


# =============================================================================
# %% get which sites are above vs below the mid-point of a conduction plane in
# ## doubleprime - without knowing where all the atoms are, as some are missing
# =============================================================================

def get_sites_above_below(plane, atoms, cell=np.eye(3), metal='Na', frac=False, viz=True):
    ## pass cell if not frac, else np.eye(3)
    
    ## get mobile ion sites
    mobile_ion_sites = get_mobile_ion_sites(atoms, plane, cell if not frac else np.eye(3), viz=False)
        
    # compose network of mobile-ion sites, calculate paths. Both beta and beta"
    # This can create and save a figure when viz=True. Pass cell if fractional was false
    _, edges, _ = get_mid_oxygen_sites_freud(mobile_ion_sites, cell if not frac else np.eye(3), viz=viz)
    nxg = nx.from_edgelist(edges)
    paths = path_lengths(nxg)
    
    ## find the ions just-above and just-below this plane, and their z-coords
    ## assume no need to wrap around boundaries in z
    zs = sorted( get_conduction_planes(atoms, metal) , key = lambda x : abs(x-plane))[:2]
    z_below = min(zs) 
    z_above = max(zs) 
    
    mobile_ions = atoms.query(f'atom == "{metal}" & {z_below-1e-3} < z < {z_above+1e-3}')
    one_ion = mobile_ions.iloc[0][['x', 'y', 'z']]
    
    ## quick query for the closest site to one mobile ion
    box = freud.box.Box.from_matrix(cell)
    query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=False)
    que = freud.locality.AABBQuery(box, mobile_ion_sites)
    first_ions_site = list(que.query(one_ion.values,query_args))[0][1]
    
    site_is_above = abs(one_ion.z-z_below) > abs(one_ion.z - z_above)
    
    modified_sites = list()
    for i, site in enumerate(mobile_ion_sites):
        site[-1] = [z_below, z_above][paths[i,first_ions_site] % 2 == site_is_above]
        modified_sites.append(site)
        
    sites_below = np.asarray([x for x in modified_sites if x[-1] == z_below])
    sites_above = np.asarray([x for x in modified_sites if x[-1] == z_above])
    
    return np.asarray(modified_sites), sites_below, sites_above, paths
    
# =============================================================================
# %% return Mg atoms near a plane with wrapping over cell boundary in z
# =============================================================================

def get_nearby_atoms_wrap(plane, atoms, distance, cell):
    
    nearby_atoms = atoms.query(f'{plane - distance} < z < {plane + distance}')
    # print(f'found nearby atoms: {len(nearby_atoms)}')
    
    ## accout for wrapping around the cell boundary
    if cell*0.5 - abs(plane) < distance:
        nearby_atoms2 = atoms.query(f'z < {-cell + abs(plane) + distance} or z > {cell - abs(plane) - distance}')
        # print(f'found wrapped Mg atoms: {len(nearby_mg2)}')
        nearby_atoms = pd.concat([nearby_atoms, nearby_atoms2]).drop_duplicates()
    
    return nearby_atoms

# =============================================================================
# %% return all Cartesian distances from sites to defects, 
# ## and count defects next to each site. This works for beta"
# =============================================================================

def get_mg_distances(site_pts, paths, defect_coords, cell=np.eye(3), frac=False, 
                     verbose=False):
    
    num_sites = len(site_pts)
    
    ## get sites closest to Mg defects. list(set()) suppresses duplicates
    edges_0, edges_1, d0, d1 = get_nearest_points(site_pts, defect_coords, cell if not frac else np.eye(3))
    sites_0 = [x[1] for x in edges_0]; counts_0 = list()
    sites_1 = [x[1] for x in edges_1]; counts_1 = list()
    
    ## calculate paths to Mg defects for all sites in this plane
    if len(sites_0) > 0:
        counts_0 = [[sites_0.count(x) for x in range(num_sites)].count(i) for i in range(6)]
        paths_to_mg0 = [min(paths[sites_0, x]) for x in range(num_sites)]
    else: paths_to_mg0 = np.ones(num_sites)*num_sites
    if len(sites_1) > 0:
        counts_1 = [[sites_1.count(x) for x in range(num_sites)].count(i) for i in range(6)]
        paths_to_mg1 = [min(paths[sites_1, x])+1 for x in range(num_sites)]
    else: paths_to_mg1 = np.ones(num_sites)*num_sites
        
    # combine path lengths to distance==1 and distance==0 sites taking min()
    paths_to_mg = [min(paths_to_mg0[i], paths_to_mg1[i]) for i in range(num_sites)]
    counts = [(sites_1+sites_0).count(x) for x in range(num_sites)]

    ## print some distances and counts of defects next to sites
    if verbose: 
        disp_counts = [counts.count(i) for i in range(6)]
        print(f'Numbers of closest defects at 0: {counts_0}')
        print(f'Numbers of closest defects at 1: {counts_1}')
        print(f'Combined #s of closest defects : {disp_counts}')
        print(f'distances at 0: {np.unique(np.round(d0,4))}')
        print(f'distances at 1: {np.unique(np.round(d1,4))}')

    return paths_to_mg, counts

# =============================================================================
# %% read a trajectory snapshot
# =============================================================================

def read_traj(file_in, num_frames=1, skip=0):
    '''
    Read a snapshot of a LAMMPS trajectory and output it as a pandas DataFrame 
    with averaging if necessary. 

    Parameters
    ----------
    file_in : string
        Path to the file with the trajectory.
    num_frames : int, optional
        Number of snapshots to average from the trajectory. The default is 1.
    skip : int, optional
        Number of snapshots to skip at the start. The default is 0.

    Returns
    -------
    atoms : pandas DataFrame
        DataFrame with real-space atom coordinates as 'x', 'y', 'z'.
    cell : numpy array, shape (3,3)
        Real-space dimensions of the cell.
    atoms_err : pandas DataFrame
        DataFrame with standard deviations of the real-space atom coordinates 
        if the latter got averaged.

    '''
    
    ## defaults
    ts = 0
    time = 0
    num_atoms = None
    cell = np.eye(3)
    atoms = list()
    atoms_err = None
    if num_frames > 1 : 
        min_std = np.zeros(num_frames)
        min_std[-1] = 1
        min_std = np.std(min_std)
    
    ## read snapshot by snapshot
    with open(file_in,'r') as fin :
        
        line = fin.readline()
        
        while line :
        
            if 'TIMESTEP' in line: 
                time = eval(fin.readline())
                ts += 1
            elif 'NUMBER OF ATOMS' in line:
                num_atoms = eval(fin.readline())
            elif 'BOX BOUNDS' in line:
                ## read cell dimensions - orthogonal cells
                for dim in range(3):
                    arr = fin.readline()[:-1].split(' ')
                    cell[dim, dim] = eval(arr[1]) - eval(arr[0])
            elif 'ATOMS' in line:
                ## parse this line for columns
                arr = line[:-1].split(' ')
                columns = arr[2:]
                # print(columns)
                
                assert num_atoms is not None, 'number of atoms not set before atoms appeared'
                
                ## read all atoms one by one, and save if not skipping this frame
                for i in range(1,num_atoms+1) :
                    arr = [eval(x) for x in fin.readline()[:-1].split(' ')]
                    if ts > skip : atoms.append(dict(zip(columns,arr)))
            
                ## stop after one snapshot according to input flag
                if num_frames + skip <= ts : break
                elif not ts % 10 : print(f'done timestep {ts}: {time}')
            line = fin.readline()
    
    ## make atoms into a dataframe
    atoms = pd.DataFrame(atoms)
                
    ## take an average if needed: this is not actually working because of boundary wrapping
    ## TODO: keep track of errors there as well
    if num_frames > 1:
        atoms_err = atoms.groupby('id').agg({'type':'first', 'xs':'std', 'ys':'std','zs':'std'}).reset_index()
        # print(atoms_err.columns)
        atoms_off = atoms_err.query(f'xs > {min_std} | ys > {min_std} | zs > {min_std}').id.unique()
        print(f'atoms crossing boundaries: {len(atoms_off)}, wrapping ...')
        
        wrapped = list()
        stds = list()
        cube = freud.box.Box.cube(1)
        for atom_id in atoms_off :
            pts = atoms.query(f'id == {atom_id}').loc[:,['xs', 'ys', 'zs']]
            # print(pts.head())
            pts = pts.values - 0.5
            # print(pts[:5,:])
            pt0 = pts[0]
            que = freud.AABBQuery(cube, pt0.reshape((1,3)))
            query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=True)
            result = list(que.query(pts[1:], query_args))
            disp_vectors = np.asarray([pts[x[1]] - pt0 for x in result])
            disp_vectors = cube.wrap(disp_vectors)
            # print(atoms_err.query(f'id == {atom_id}').type.iloc[0])
            wrapped.append(dict(atom_id=atom_id, xs=pt0[0]+np.mean(disp_vectors[:,0])+0.5,
                                ys=pt0[1]+np.mean(disp_vectors[:,1])+0.5,
                                zs=pt0[2]+np.mean(disp_vectors[:,2])+0.5,
                                atom_type=atoms_err.query(f'id == {atom_id}').type.iloc[0]))
            stds.append(dict(atom_id=atom_id, xs=np.std(disp_vectors[:,0]),
                                ys=np.mean(disp_vectors[:,1]),
                                zs=np.mean(disp_vectors[:,2]),
                                atom_type=atoms_err.query(f'id == {atom_id}').type.iloc[0]))
        
        wrapped = pd.DataFrame(wrapped).rename(columns={'atom_id':'id', 'atom_type':'type'})
        stds = pd.DataFrame(stds).rename(columns={'atom_id':'id', 'atom_type':'type'})
        atoms_err = atoms_err.query(f'id not in {list(atoms_off)}').append(stds, ignore_index=True).sort_values(by='id').set_index('id')
        
        ## check
        new_atoms_off = atoms_err.query(f'xs > {min_std} | ys > @min_std | zs > @min_std').index.values
        print(f'now atoms crossing boundaries: {len(new_atoms_off)}')
        
        ## merge wrapped atoms to full dataframe
        atoms = atoms.groupby('id').agg({'type':'first', 'xs':'mean', 'ys':'mean','zs':'mean'}).reset_index()
        atoms = atoms.query(f'id not in {list(atoms_off)}').append(wrapped, ignore_index=True).sort_values(by='id').set_index('id')
        # print(atoms.columns)
        
    ## check completeness
    
    ## multiply dimensions to return real-space coordinates
    for i in range(3):
        atoms[dims[i]] = atoms[f'{dims[i]}s'] * cell[i,i]
        atoms.drop(f'{dims[i]}s', axis=1, inplace=True)
        
        
        if num_frames > 1 :
            atoms_err[dims[i]] = atoms_err[f'{dims[i]}s'] * cell[i,i]
            atoms_err.drop(f'{dims[i]}s', axis=1, inplace=True)
    
    return atoms, cell, atoms_err

# =============================================================================
# %% assign symmetry types to atoms for beta:
## for beta, make a new column with atom types by symmetry (Al, O) & site(mobile)
## Al1 : exactly between planes, not above/below BR sites (Oh symmetry)
## Al2 : almost exactly between planes, directly above/below O5 (Td)
## Al3 : just above & below conduction planes (Oh)
## Al4 : at the in-plane oxygens (Td)
## Al5 : at the Roth defects

## O1 : closest to midpoints between planes - O2s are a subset of O1
## O2 : subset of O1 next to Al2
## O3 : just above & below conduction planes at aBR sites
## O4 : just above & below conduction planes at BR sites
## O5 : in-plane oxygens
## Oi : interstitial oxygens
# =============================================================================

def add_symmetry_beta(atoms, cell=np.eye(3), mobile='Na', frac=False):
    
    ## create the column
    atoms['symm'] = 'a'
    
    ## add id column if it is not there
    if 'id' not in atoms.columns : atoms['id'] = atoms.index.values
    
    mobile_ions = atoms.query('atom==@mobile').sort_values(by='z')
    num_planes = round(cell[2,2] / mobile_ions.z.diff().dropna().max())
    
    ## it would be simple to extend the plane detection to unequal #s of atoms
    st = len(atoms.query('atom==@mobile')) // num_planes
    plane_zs = [mobile_ions.iloc[st*x:st*(x+1)].z.mean() for x in range(num_planes) ]
    dz = (max(plane_zs) - min(plane_zs)) / (num_planes-1)
    
    # print(st, plane_zs)
    
    ## create a column with by-atom distances from a plane
    atoms['dz'] = atoms.z.apply(lambda x: min([abs(x-y) for y in plane_zs]))/dz
    
    ## assign Al4, then Al3, then Al1+Al2 (Al2 is assigned later)
    atoms.loc[atoms.atom == 'Al', 'symm'] = 'Al4'
    atoms.loc[(atoms.dz > 0.212) & (atoms.atom == 'Al'), 'symm'] = 'Al3'
    atoms.loc[(atoms.dz > 0.4) & (atoms.atom == 'Al'), 'symm'] = 'Al1'
    
    ## assign O5, then O4+O3, then O1+O2 (O3 and O2 assigned later)
    atoms.loc[atoms.atom == 'O', 'symm'] = 'O5'
    atoms.loc[(atoms.atom == 'O') & (atoms.dz > 0.144), 'symm'] = 'O4'
    atoms.loc[(atoms.atom == 'O') & (atoms.dz > 0.288), 'symm'] = 'O1'
    
    ## assign Oi
    if 'type' in atoms.columns :
        atoms.loc[(atoms.type == 4) & (atoms.atom == 'O'), 'symm'] = 'Oi'
    
    ## assign Mg if they exist
    atoms.loc[atoms.atom == 'Mg', 'symm'] = 'Mg'
    
    ## for each plane: find the aBR and BR sites, find defects,  
    ## and assign O3, O2, Al2, and sites to mobile ions
    for p in plane_zs:
        o5s = atoms.query(f'symm == "O5" & {p-dz/3} < z < {p+dz/3}')
        o4s = atoms.query(f'symm == "O4" & {p-dz/2} < z < {p+dz/2}'); # print(len(o4s))
        o1s = atoms.query(f'symm == "O1" & {p-dz/2} < z < {p+dz/2}');
        ois = atoms.query(f'symm == "Oi" & {p-dz/3} < z < {p+dz/3}')
        a1s = atoms.query(f'symm == "Al1" & {p-dz} < z < {p+dz}')
        mis = atoms.query(f'atom == @mobile & {p-dz/3} < z < {p+dz/3}'); mis.z=0
        
        # print(len(o5s))
        mobile_sites = get_mobile_ion_sites(o5s, p, cell, thresh=100)
        # print(f'found {len(mobile_sites)} mobile-ion sites at z={p:.3f}')
        
        ## get mid-oxygen sites
        _, edges, midpts = get_mid_oxygen_sites_freud(mobile_sites, cell, viz=False)
        # print(edges[:5], midpts[:5])
        
        ## calculate where the Oi's are by distance, call those edges 'picked'
        picked = list()
        for i, oi in ois.iterrows():
            r2 = (midpts[:,0]-oi.x)**2 + (midpts[:,1]-oi.y)**2
            # print(np.argwhere(r2 == min(r2))[0][0])
            picked.append(edges[np.argwhere(r2 == min(r2))[0][0]])
        picked = set(picked)
        sites_next_to_oi = np.array(sorted([x for x in flatten(list(picked))]))
        
        ## create a proper networkx graph from edge list, and calculate path lengths
        nxg = nx.from_edgelist(edges)
        paths = path_lengths(nxg)
        
        ## measure all path lengths to the oxygens; this yields a list
        ## if there are no such paths - then there are no defects
        if picked :
            paths_to_oi = [min(paths[sites_next_to_oi,x]) for x in range(len(mobile_sites))]
        else :
            paths_to_oi = [100 for x in range(len(mobile_sites))]
            
        
        ## get BR and a-BR sites for labeling mobile ions
        site_types = auto_get_BR_sites(atoms.query('symm != "Oi"'), cell, mobile_sites, frac)
        # print(sum([x=='BR' for x in site_types]), 'BR')
        # print(sum([x=='aBR' for x in site_types]), 'aBR')
        # print([x for x in site_types if 'error' in x])
        # print(sum([x=='doubleprime' for x in site_types]), 'bdp')
        
        ## try making a lattice of BR's that would cover all BR's
        BR_sites = list(); aBR_sites = list()
        for br in [x for x in range(len(site_types)) if site_types[x] == 'BR'] :
            BR_sites = [x for x in range(len(site_types)) if site_types[x] == 'BR' and not paths[br,x] % 2]
            if len(BR_sites) == sum([x=='BR' for x in site_types]) :
                BR_sites  = [x for x in range(len(site_types)) if not paths[br,x] % 2]
                aBR_sites = [x for x in range(len(site_types)) if paths[br,x] % 2]
                # print(len(BR_sites),len(aBR_sites))
                break
            ## some safeguard or error message needs to go here if this fails
        BR_locs = mobile_sites[BR_sites]
        BR_locs[:,-1] = p
        aBR_locs = mobile_sites[aBR_sites]
        aBR_locs[:,-1] = p
        
        ## find in which site each mobile ion resides using freud AABBQuery
        # create the freud query (freud 2.2.0)
        query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=False)
        box = freud.box.Box(Lx=cell[0,0], Ly=cell[1,1], is2D=True)
        mobile_sites[:,-1] = 0
        que = freud.locality.AABBQuery(box, mobile_sites) 
        
        result = que.query(mis[['x','y','z']].values, query_args)
        
        ## assign distance+site for every mobile ion
        for r in result:
            ## first item should be ion, second should be site
            atomid = mis.iloc[r[0]].id
            siteid = r[1]
            atoms.loc[atoms.id == atomid, 'symm'] = str(int(paths_to_oi[siteid])) + ('A' if siteid in aBR_sites else 'B')
        
        ## assign O3: query above/below aBR sites
        query_args2 = dict(mode='nearest', num_neighbors=2, exclude_ii=False)
        box2 = freud.box.Box(Lx=cell[0,0], Ly=cell[1,1], Lz=cell[2,2])
        que2 = freud.locality.AABBQuery(box2, o4s[['x','y','z']].values)
        result2 = que2.query(aBR_locs, query_args2)
        
        ## actually change the symmetry
        # o3s = 0
        for r in result2:
            atomid = o4s.iloc[r[1]].id
            atoms.loc[atoms.id == atomid, 'symm'] = 'O3'
            # o3s += 1
        # print(o3s)
        
        ## assign Al2: above / below O5's
        que3 = freud.locality.AABBQuery(box2, a1s[['x','y','z']].values)
        result3 = que3.query(o5s[['x','y','z']].values, query_args2)
        for r in result3:
            atomid = a1s.iloc[r[1]].id
            atoms.loc[atoms.id == atomid, 'symm'] = 'Al2'
        
        ## assign O2: above/below BR sites
        que4 = freud.locality.AABBQuery(box2, o1s[['x','y','z']].values)
        result4 = que4.query(BR_locs, query_args2)
        for r in result4:
            atomid = o1s.iloc[r[1]].id
            atoms.loc[atoms.id == atomid, 'symm'] = 'O2'
    
    return atoms.drop(columns='dz')
    
# =============================================================================
# %% write a CIF file from a snapshot
# =============================================================================

def write_cif(filename, atoms, cell=np.eye(3), frac=False):
    
    ## add symmetry elements
    atoms = add_symmetry_beta(atoms, cell, frac=frac)
    
    ## make atoms fractional and 0-1 rather than -0.5 to 0.5
    for i, dim in enumerate(['x', 'y', 'z']) :
        cell[i,i] = atoms[dim].max() - atoms[dim].min()
        atoms[dim] = (atoms[dim]-atoms[dim].min()) / (atoms[dim].max() - atoms[dim].min())
    
    ## Open an output file. This will overwrite if a file with this name exists.
    fout = open(filename, 'w')

    ## Write some front matter
    fout.write('#=====\n#generated with python\n#=====\ndata_snapshot\n\nloop_\n')
    
    ## write compound formula
    formula = ' '.join([x + str(len(atoms.query(f'atom == "{x}"'))) for x in atoms.atom.unique()])
    fout.write(f'{"_chemical_name_common":<40}\'{formula}\'\n')

    ## write the dimensions of the simulation box
    for n, dim in zip(['a','b','c'], [0,1,2]):
        fout.write(f'_cell_length_{n:<27}{cell[dim,dim]:.4f}\n')
    
    ## write angles, works only for orthogonal now
    for ang in ['alpha', 'beta', 'gamma']:
        fout.write(f'_cell_angle_{ang:<28}90\n')
        
    ## write stuff like symmetry elements - not implemented
        
    ## write columns for atoms
    fout.write('\nloop_\n')
    props = ['_atom_site_label', '_atom_site_occupancy', '_atom_site_fract_x',
             '_atom_site_fract_y', '_atom_site_fract_z', '_atom_site_type_symbol']
    for prop in props:
        fout.write(prop+'\n')
    
    ## write atoms 
    for i, a in atoms.iterrows():
        fout.write(f'{a.atom:<3}1.0  {a.x:.6f} {a.y:.6f} {a.z:.6f} {a.symm}\n')
        if not i % 500 : print(f'wrote atom #{i}')
    
    ## write charges
    fout.write('\nloop_\n')
    for prop in ['_atom_type_symbol', '_atom_type_oxidation_number'] :
        fout.write(prop+'\n')
        
    for s in sorted(atoms.symm.unique()) :
        ## default is the mobile ion
        chg = 1.0
        if 'Al' in s : chg = 3.
        elif 'Mg' in s : chg = 2.
        elif 'O'  in s : chg = -2.
        fout.write(f'{s:<4} {chg:.3f}\n')
    
# =============================================================================
# %% assign symmetry types to atoms for beta-doubleprime:
## for beta, make a new column with atom types by symmetry (Al, O) & site(mobile)
## Al1 : second layer of Al from planes (Oh symmetry)
## Al2 : almost exactly between planes (Td)
## Al3 : at the in-plane oxygens (Td)
## Al4 : exactly between planes (Oh)

## O1 : just above & below conduction planes, but not directly above/below sites
## O2 : same plane as O4, just not directly above / below Al2 and not next to Al4
## O3 : just above & below conduction planes, directly above/below sites and further from O5
## O4 : directly above / below Al2
## O5 : in-plane oxygens
# =============================================================================

def add_symmetry_bdp(atoms, cell=np.eye(3), mobile='Na', frac=False):
    
    ## create the column
    atoms['symm'] = mobile
    
    ## add id column if it is not there
    if 'id' not in atoms.columns : atoms['id'] = atoms.index.values
    
    mobile_ions = atoms.query('atom==@mobile').sort_values(by='z')
    num_planes = round(cell[2,2] / mobile_ions.z.diff().dropna().max())
    
    ## it would be simple to extend the plane detection to unequal #s of atoms
    st = int(len(atoms.query('atom==@mobile')) / num_planes)
    plane_zs = [mobile_ions.iloc[st*x:st*(x+1)].z.mean() for x in range(num_planes) ]
    dz = (max(plane_zs) - min(plane_zs)) / (num_planes-1)
    
    # print(f'{st} mobile ions per plane at zs: {plane_zs}')
    
    ## create a column with by-atom distances from a plane
    atoms['dz'] = atoms.z.apply(lambda x: min([abs(x-y) for y in plane_zs]))/dz
    
    ## assign Als by increasing distance : Al3, then Al1, then Al2, then Al4
    atoms.loc[atoms.atom == 'Al', 'symm'] = 'Al3'
    atoms.loc[(atoms.dz > 0.205) & (atoms.atom == 'Al'), 'symm'] = 'Al1'
    atoms.loc[(atoms.dz > 0.38) & (atoms.atom == 'Al'), 'symm'] = 'Al2'
    atoms.loc[(atoms.dz > 0.475) & (atoms.atom == 'Al'), 'symm'] = 'Al4'

    ## check the Al assignments
    # print(atoms.query('atom == "Al"').groupby('symm').agg('count'))
    
    # print(sorted(atoms.query('atom == "Al"').dz.unique()))

    ## assign O5, then O1+O2 (O3 and O4 assigned later)
    atoms.loc[atoms.atom == 'O', 'symm'] = 'O5'
    atoms.loc[(atoms.atom == 'O') & (atoms.dz > 0.1), 'symm'] = 'O1' ## this has O3
    atoms.loc[(atoms.atom == 'O') & (atoms.dz > 0.285), 'symm'] = 'O2' ## this has O4
    
    ## assign Oi
    if 'type' in atoms.columns :
        atoms.loc[(atoms.type == 4) & (atoms.atom == 'O'), 'symm'] = 'Oi'
    
    ## assign Mg if they exist
    atoms.loc[atoms.atom == 'Mg', 'symm'] = 'Mg'
    
    ## assign O4 : next to Al4
    # print('starting query for assigning O4')
    o2s = atoms.query('symm == "O2"')
    al4s = atoms.query('symm == "Al4"')[['x','y','z']].values
    
    box = freud.box.Box(Lx=cell[0,0], Ly=cell[1,1], Lz=cell[2,2])
    query_args = dict(mode='nearest', num_neighbors=6, exclude_ii=False)
    que_O4 = freud.locality.AABBQuery(box, al4s) 
    result_O4 = np.array(list(que_O4.query(o2s[['x','y','z']].values, query_args)))
    
    idxs_O4 = list(o2s.iloc[result_O4[:,1]].id.values.astype(int))
    atoms.loc[idxs_O4, 'symm'] = 'O4'
    
    ## assign O3: the O1's far away from O5's at distances > 3.5 angstroms
    ## this does not work for some reason
    o1s = atoms.query('symm == "O1"')
    # o5s = atoms.query('symm == "O5"')[['x','y','z']].values
    
    # box = freud.box.Box(Lx=cell[0,0], Ly=cell[1,1], Lz=cell[2,2])
    query_args_O3 = dict(mode='nearest', num_neighbors=1, exclude_ii=False)
    # que_O3 = freud.locality.AABBQuery(box, o5s) 
    # result_O3 = np.array(list(que_O3.query(o1s[['x','y','z']].values, query_args_O3)))
    
    # idxs_O3 = set(list(o1s.iloc[result_O3[:,1]].id.values.astype(int)))
    # print(f'Found {len(list(idxs_O3))} O3s ')
    # atoms.loc[list(idxs_O3), 'symm'] = 'O3'
    
    ## assign O3 as being the closest O to mobile-ion sites
    for z in plane_zs : 
        
        o1s = atoms.query(f'symm == "O1" & {z -2.5} < z < {z+2.5}')
        sites = get_mobile_ion_sites(atoms, z, cell, viz=False, thresh=1)
        que_O3 = freud.locality.AABBQuery(box, sites)
        result_O3 = np.array(list(que_O3.query(o1s[['x','y','z']].values, query_args_O3)))
        
        idxs_O3 = set(list(o1s.iloc[result_O3[:,1]].id.values.astype(int)))
        # print(f'Found {len(list(idxs_O3))} O3s at plane {z:.3f}')
        # print(set(sites[:,2]))
        atoms.loc[list(idxs_O3), 'symm'] = 'O3'
        # print(atoms.query('symm == "O3"').z.round(2).unique())
    
        # ## check the O symmetry assignments
        # print(atoms.groupby('symm').agg('count'))
    
    return atoms.drop(columns=['id','dz'])




























