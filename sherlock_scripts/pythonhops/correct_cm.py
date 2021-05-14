#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 2020

Corrects the mobile-ion center of mass trajectory to be in the reference frame
of the host lattice. Following Marcolongo & Marzari (2017)

Command-line options required:
    - atoms  : crystal structure
    - template : prefix to the filenames of the center of mass fixes
    
Assumes that the columns are x,y,z,vx,vy,vz.

@author: andreypoletaev
"""

from sys import argv
import numpy as np

from glob import glob

from crystal_utils import read_lmp
from crystal_utils import write_masses as masses

from hop_utils import which_one_in

from pandas import read_csv 

mobile_ions = ['Na', 'Ag', 'K']
host_ions = ['Al', 'O', 'Mg']

## flag, True if correcting velocities
vel = False

## placeholder for the name of the output file
fout_name = None

# =============================================================================
# %% read the first few lines of a trajectory and save the cell dimensions
# =============================================================================

## Parse inputs. Format: key=value
options = dict([ (x.split('=')[0], x.split('=')[1]) for x in argv[1:] ])
keys = list(options.keys())

assert 'atoms'  in keys, 'pass atoms=...  [path] as a command-line option'
assert 'template' in keys, 'pass template=... [path] as a command-line option'
# assert 'file_out' in keys, 'pass file_out=... [path] as a command-line option'

if 'velocity' in keys and options['velocity'] == 'True' :
    vel = True
    print('Velocities will be included')

_, _, cell, atoms = read_lmp(options['atoms'], fractional=False)
mm = which_one_in(mobile_ions, atoms.atom.unique())
atoms['mass'] = [masses[x] for x in atoms.atom]
cms = atoms.groupby('atom').agg('sum')
cm_masses = dict([(atom, cms.loc[atom,'mass']) for atom in cms.index.values])
m_total = cms.mass.sum()
m_host   = cms.reset_index().query(f'atom != "{mm}"').mass.sum()
m_mobile = cms.reset_index().query(f'atom == "{mm}"').mass.iloc[0]

print(f'atoms: {atoms.atom.unique()}')

## find all the host atoms and their tCoM masses
host_file_dict = dict()
for atom in atoms.atom.unique() :
    fin_names = glob(f'{options["template"]}-{atom}.fix')
    if atom == mm :
        ## open the file for the mobile-ion CoM
        fin = open(fin_names[0], 'r')
        head1 = fin.readline()
        head2 = fin.readline()
        print(f'mobile-ion file: {fin_names[0]}')
        
        ## open a new file for the corrected CoM
        fout_name = fin_names[0][:-4]+'-cor.csv'
        
        try :
            partial = read_csv(fout_name)
            last_done_time = partial.time.values[-1]
            print(f'Found a partially done file with last time {last_done_time} ps.')
            fout = open(fout_name,'a')
        except :
            print(f'Found no pre-done file, starting a new one: {fout_name}')
            last_done_time = 0
            fout = open(fout_name,'w')
            fout.write('time,x,y,z\n' if not vel else 'time,x,y,z,vx,vy,vz\n')
    else :
        host_file_dict[atom] = open(fin_names[0], 'r')
        host_file_dict[atom].readline()
        host_file_dict[atom].readline()
        print(f'host file: {fin_names[0]}')
        
## first lineof the mobile-ion file
line = fin.readline()

## read a line from each of the files
lines = [(host_file_dict[atom].readline(), cm_masses[atom]) for atom in host_file_dict.keys()]

while line and lines:
    ## make an array for the corrected CoM position & velocity
    mobile_arr = np.array(line[:-1].split(' ')).astype(float)
    time = np.round(mobile_arr[0]/1000,3)
    
    if time <= last_done_time : 
        ## read the next line for the mobile ion & host lattice ions
        line = fin.readline()
        lines = [(host_file_dict[atom].readline(), cm_masses[atom]) for atom in host_file_dict.keys()]
        continue
    
    ## initialize a data structure to hold host-lattice displacements
    host_cm = np.zeros((len(host_file_dict.keys()),3))
    
    ## process each line, taking x,y,z
    for i, (host_line, cm_mass) in enumerate(lines) :
        x_arr = np.array(host_line[:-1].split(' ')[1:4]).astype(float)
        # v_arr = np.array(host_line[:-1].split(' ')[4:7]).astype(float)
        
        host_cm[i,:] = x_arr * cm_mass
        
    host_cm = np.sum(host_cm, axis=0) / m_host
    x_arr = mobile_arr[1:4] - host_cm
    
    if vel :
        ## initialize a data structure to hold host-lattice velocities
        host_mv = np.zeros((len(host_file_dict.keys()),3))
        ## process each line, taking x,y,z
        for i, (host_line, cm_mass) in enumerate(lines) :
            # x_arr = np.array(host_line[:-1].split(' ')[1:4]).astype(float)
            v_arr = np.array(host_line[:-1].split(' ')[4:7]).astype(float)
            
            host_mv[i,:] = v_arr * cm_mass
        host_mv = np.sum(host_mv, axis=0) / m_mobile
        
        ## make an array for the corrected CoM velocity, and concat w/ position
        v_arr = mobile_arr[4:7] - host_mv
        x_arr = np.concatenate((x_arr,v_arr), axis=None)
    
    ## write corrected CoM 
    fout.write(str(time)+','+','.join(np.round(x_arr,6).astype(str))+'\n')
    
    ## autosave progress and restart the output file
    if not time % 5000 : 
        fout.close()
        fout = open(fout_name,'a')
        print(f'Autosaved and restarted output at time {time} ps.', flush=True)
        
    ## write a progress output
    if not time % 500 : print(f'completed {time} psec.')
    
    ## read the next line for the mobile ion & host lattice ions
    line = fin.readline()
    lines = [(host_file_dict[atom].readline(), cm_masses[atom]) for atom in host_file_dict.keys()]
    
fin.close()
fout.close()

for k,v in host_file_dict.items() :
    v.close()

# with open(options['file_in'], 'r') as fin:
    
#     for i in range(5) : l = fin.readline()
    
#     cell = eye(3)
    
#     for i in range(3) :
#         l = fin.readline()
#         d = array(l[:-1].split(' ')).astype(float)
#         cell[i,i] = max(d) - min(d)
        
#     savetxt(options['file_out'], cell, delimiter=' ', fmt='%.7g')