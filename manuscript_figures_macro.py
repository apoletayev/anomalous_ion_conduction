#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andreypoletaev
"""

# =============================================================================
# %% Block 1: initial imports
# =============================================================================

import os, sys, re, glob

if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import numpy as np
import pandas as pd
import hop_utils as hu

from crystal_utils import read_lmp
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from batlow import cm_data
batlow_cm = LinearSegmentedColormap.from_list('batlow', cm_data)
batlow_even = LinearSegmentedColormap.from_list('batlow_even', hu.batlow_lightness_scaled(0.4,0.6))

from cycler import cycler 
linecycler = cycler(linestyle=['-', '--', ':', '-.'])
markcycler = cycler(marker=['o', 's', 'v', 'd', '^'])

from itertools import cycle
markers = cycle(['o', 's', 'v', 'd', '^','D','<','>']) 
lines = cycle(['-', '--', '-.', ':'])

## linear fitting
linfit = lambda x, *p : p[0] * x + p[1]

## cosmetic defaults for matplotlib plotting 
plt.rc('legend', fontsize=10)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('errorbar', capsize=3)
plt.rc('markers', fillstyle='none')
plt.rc("savefig", format='pdf')

## variables by which it is possible to plot
relevant_vars = ['metal','phase','T1','config','stoich','exclude','z']

## which atoms to query for species

## conductivity from bulk diffusion coefficient. Takes D_bulk [cm^2/sec], cell [AA]
## output is [Kelvin/ohm/cm] i.e. [Kelvin * siemens / cm]
## note that there is only one q in the formula because hu.kb is [eV/K]
q = 1.602e-19   ## [Coulomb] elementary charge
AA = 1e-8       ## [cm] 1 angstrom in cm
sigma_T = lambda N, cell, d_com : q * N / np.prod(np.diag(cell*AA))*d_com / hu.kb 

unit_conv = 1e-4 ## [cm^2/sec] 1 AA^2/psec = 0.0001 cm^2/sec. No need to change this.
eps_0 = 8.854187e-12 ## [A^2 m^-3 kg^-1 sec^4]

T1 = 300
## dictionary of units
units = {'T1':'K', 'metal':'', 'stoich':'', 'exclude':'', 'config':'', 'z':'',
         'phase':f' {T1}K'}

## shorthands for labels
bdp =  r'$\beta^{\prime\prime}$'
beta = r'$\beta$'
phases = {'beta':beta, 'bdp':bdp}

# =============================================================================
# %% Block 2 : load files based on the index of conduction planes created in 
# ## analysis_steadystate.py
# ## The a2_...fix files are assumed to be located in the same folders as their 
# ## corresponding lammps structure files. 
# =============================================================================

## database of all the hops: only combined planes matter for macro analyses.
all_planes = pd.read_csv('./sample_data/all_hop_planes.csv').query('z == "z_all"')

## flag for loading atoms
frac = False

## flag for loading CoM immediately
load_com = False

## flag for loading the output of the LAMMPS msd fix 
load_r2 = False

## ===== BETA single-metal =====

## ===== BETA Ag =====

planes_to_load = all_planes.query('metal == "Ag" & config == "120_4" & T1 in [300,600,1000]')

## ===== BETA Na =====

# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & stoich == "120" ')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & stoich == "120" & T1 == 300')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & stoich == "120" & T1 == 600')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & config == "120_4" & T1 in [300,600,1000]')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & config == "120_4" & T1 in [300,600]')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & config == "120_4" & T1 == 300')
# planes_to_load = all_planes.query('metal == "Na" & phase == "beta" & config == "120_1"')

## ===== BETA K =====

# planes_to_load = all_planes.query('metal == "K" & stoich == "120" & 300 < T1 < 900')
# planes_to_load = all_planes.query('metal == "K" & stoich == "120" & T1 in [300, 600]')
# planes_to_load = all_planes.query('metal == "K" & config == "120_4"')
# planes_to_load = all_planes.query('metal == "K" & config == "120_4" & T1 in [300,600,1000]')

## ===== BETA all metals together =====

# planes_to_load = all_planes.query('phase == "beta" & config == "120_4" & T1 == 1000')
# planes_to_load = all_planes.query('phase == "beta" & config == "120_4" & T1 == 600')
# planes_to_load = all_planes.query('phase == "beta" & config == "120_4" & T1 in [300,600,1000] ')

## ===== BDP =====

## ===== BDP Na =====

planes_to_load = all_planes.query('phase != "beta" & metal == "Na" & config == "unsym_0" & T1 in [230,300,473,600]')
# planes_to_load = all_planes.query('phase != "beta" & metal == "Na" & config == "unsym_0" & T1 in [230,300]')
# planes_to_load = all_planes.query('phase != "beta" & metal == "Na" & config == "unsym_0" & T1 in [300,473,600]')

# planes_to_load = all_planes.query('phase != "beta" & metal == "Na" & T1 == 600')
# planes_to_load = all_planes.query('phase != "beta" & metal == "Na" & stoich in ["unsym", "unsymLi"] & T1 in [230,300,473]')
# planes_to_load.sort_values('config', ascending=False, inplace=True)

## ===== BDP K =====

# planes_to_load = all_planes.query('phase != "beta" & metal == "K"')
# planes_to_load = all_planes.query('phase != "beta" & metal == "K" & config == "symm_1"')
# planes_to_load = all_planes.query('phase != "beta" & metal == "K" & config == "unsym_0"')
# planes_to_load = all_planes.query('phase != "beta" & metal == "K" & config == "unsym_0" & T1 in [300,600]')
# planes_to_load = all_planes.query('phase != "beta" & metal == "K" & config == "unsym_0" & T1 == 300')
# planes_to_load = all_planes.query('phase != "beta" & metal == "K" & config == "unsym_0" & T1 == 600')
# planes_to_load.sort_values('config', ascending=False, inplace=True)

## ===== BDP Ag =====

# planes_to_load = all_planes.query('phase != "beta" & metal == "Ag"')
# planes_to_load = all_planes.query('phase != "beta" & metal == "Ag" & config == "symm_1"')
# planes_to_load = all_planes.query('phase != "beta" & metal == "Ag" & config == "unsym_0"')
# planes_to_load.sort_values('config', ascending=False, inplace=True)

## ===== BDP all metals together =====

# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "symm_1" & T1 == 600')
# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "symm_1" & T1 in [230,300,473,600]')
# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 == 300')
# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 == 600')
# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 in [300,600]')
# planes_to_load = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 in [230,300,473,600]')
# planes_to_load = all_planes.query('phase != "beta" & metal in ["Na", "K"] & T1 in [230,300,473,600]')
# planes_to_load = all_planes.query('phase != "beta" & T1 in [230,300,473,600]')
# planes_to_load = all_planes.query('phase != "beta" & config == "unsym_0" & metal in ["Ag", "K"] & T1 in [230,300,473,600]')
# planes_to_load.sort_values('config', ascending=False, inplace=True)

## ===== both beta and doubleprime =====

# planes_to_load = all_planes.query('metal == "Na" & T1 == 300 & config in ["120_4", "unsym_0", "symm_1", "102_1"]')

# ========== automatic things below this line ==========

## make a structure for loading data
planes_dicts = []

## load macro-analysis files from the lammps non-Gaussian compute
for plane in planes_to_load.itertuples(index=False):
    
    mm = plane.metal
    T1 = plane.T1
    hp = plane.hop_path
    ph = plane.phase
    st = plane.stoich
    ex = plane.exclude
    tt = plane.total_time
    cn = plane.config
    
    ## load lammps structure
    _, _, cell, atoms = read_lmp(plane.lammps_path, fractional=False)
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
    a2_fnames = glob.glob(a2_folder+f'/{mm}*a2-*{T1}K*ps.csv')
    
    ## load the a2 file if exactly one exists, else complain
    if a2_fnames :
        if len(a2_fnames) > 1 : a2_fnames = sorted(a2_fnames, reverse=True,
                        key = lambda x : eval(re.split('-|_| ',x)[-1][:-6]))
        
        # ## read the a2 file - options for fix file
        # this_a2 = pd.read_csv(a2_fname[0], names=['time','r2','r4','a2'],
        #                       skiprows=2, sep=' ')
        # this_a2.time /= 1000
        
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
    com_fname = glob.glob(a2_folder + f'/cm*{T1}K*{mm}.fix')
    if isinstance(com_fname, list) and len(com_fname) == 1 and load_com:
        
        this_com = pd.read_csv(com_fname[0],sep=' ', names=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'], skiprows=2).drop(columns=['vx','vy','vz'])
        this_com.time /= 1000. ## hard-coded conversion from steps to picoseconds
        this_com.set_index('time', inplace=True)
        print('Loaded CoM trajectory.')
    elif not load_com :
        this_com = True
        print('Skipping CoM trajectory.')
    else : 
        print(f'Could not load CoM trajectory, found: {com_fname}')
        this_com = None
       
    ## wrap the a2, CoM, and metadata into a dict
    if (this_r2 is not None or not load_r2) and (this_a2 is not None)  :
    # if (this_a2 is not None) and (this_r2 is not None) and (this_com is not None) :
        planes_dicts.append(dict(phase=ph, metal=mm, T1=T1, config=cn, stoich=st, exclude=ex,
                                 a2=this_a2, lit_folder=lit_folder, com = this_com,
                                 cell=cell, atoms=atoms, folder=a2_folder, r2=this_r2))
        
## make the holding structure into a dataframe    
macro_planes_data = pd.DataFrame(planes_dicts)

# =============================================================================
# %% Figure 2 (and Extended Data 1-6) : (a) r2, (b) exponent of r2 vs distance, 
# ## (c) dx raw, (d) dx rescaled,(e) Gs 2D color plot, (f) Gs fitting.
# ## Version "04", March 29 2021, this is in manuscript versions 07-09
# =============================================================================

## parameters: 
## THIS RELIES ON LOADING PLANES ABOVE
    
dim = 2             ## dimension for r2, typically just 2, but 3 is possible.
guides = True       ## plot guidelines in (a)
hop_length = 2.8    ## [AA] for binning dx
dx_times = [25, 25e1, 25e2, 25e3]   ## time points for spectra of dx
gs_times = 2.5*np.logspace(-1,4,6) ## [ps] times for plotting spectra
rs_list  = [[0.01,1.7], [0.01, 4.6]] ## use 1.7/4.6 for bdp, 1.6/4.3 for beta?

T1_dx = 300     ## for which temperature to plot dx
T1s_gs = [300,600]     ## for which temperatures to plot Gs fits

cd_exclude = []

na_bdp_unsym = True  ## trigger for making a broken axis for C_D(t) (Figure 2)

# ========== automatic things below this line ==========

## parameters to transform the C_D and create the broken axes
cd_break = 4.25
cd_break_top = 4.6
cd_scale = 2. # linear scaling factor; if > 1 makes transform look compressed
cd_true = 9  # value that will show up as cd_display 
cd_display = 6.8 # value at which in 

# linear function that calculates an underlying transformed coordinate 
# given y points from real data. 
# maps cd_true to cd_display; true y-value at cd_true shows up as at cd_display
cd_transform = lambda y : cd_display + (y - cd_true) / cd_scale

## new figure & counter for Gs fitting colors
fig, axes = plt.subplots(3,2, figsize=(10,12))

## make a color map for all temperature values
T1_colors = cycle([batlow_even(i) for i in np.linspace(0, 1, len(macro_planes_data.T1.unique()))])
dx_colors = cycle([batlow_even(i) for i in np.linspace(0, 1, len(dx_times))])

## one figure total
for i, plane in macro_planes_data.iterrows():
    
    mm = plane.metal; st = plane.stoich; cn = plane.config; ph = plane.phase
    ex = plane.exclude; T1 = plane.T1; folder = plane.folder
    
    cc = next(T1_colors)
    
    ## set a legend title, and plotting labels for each curve
    label = str(T1) + 'K'
    leg_title = f'{mm} {phases[ph]}'
    
    ## load the 2D a2 file - leaving out the "split" files
    if dim == 2 :
        a2_xys = glob.glob(folder+f'/{mm}*{ex}-a2xy*{T1}K*ps.csv')

        ## load the a2 file if exactly one exists, else complain
        if a2_xys :
            if len(a2_xys) > 1 : a2_xys = sorted(a2_xys, reverse=True,
                            key = lambda x : eval(re.split('-|_| ',x)[-1][:-6]))

            ## read the a2 file - options for csv file
            a2 = pd.read_csv(a2_xys[0], sep=',').set_index('time')
        else : print(f'could not load a 2D a2 file for plane {mm} {cn} {T1}K')
    else : a2 = plane.a2
        
    ## recalculate a2 for the right number of dimensions
    a2.a2 = dim * a2.r4 / a2.r2 ** 2 / (dim+2) - 1
    
    ## load a short-time a2 file if using one
    try : 
        a2s = pd.read_csv(folder + f'/{mm}-{st}-{ex}-a2{"xy" if dim == 2 else ""}-{T1}K-10ps.csv').set_index('time')
        a2s.a2 = dim * a2s.r4 / a2s.r2 ** 2 / (dim+2) - 1
    except : a2s = None
    
    # ===== (a) r2 =====
    axes[0,0].plot(a2.r2.iloc[1:], label=label, c=cc)
                    
    axes[0,0].legend(title=leg_title)
    
    if guides and T1 == T1_dx :
        if mm == 'Na' and 'unsym' in cn :
            axes[0,0].plot([0.03,0.03*5], [0.07,0.07*25], c='k', lw=0.4)
            axes[0,0].plot([15e2, 15e3], [2e3, 2e4], c='k', lw=0.4)
            axes[0,0].plot([4e3, 4e4], [4, 4*10**0.75], c='k', lw=0.4)
        elif mm == 'Na' and 'symm' in cn :
            axes[0,0].plot([15e2, 15e3], [2e3, 2e4], c='k', lw=0.4)
            axes[0,0].plot([25e2, 25e3], [2e1, 2e2], c='k', lw=0.4)
        elif mm == 'Na' and '120_4' in cn :
            axes[0,0].plot([0.03,0.03*5], [0.12,0.12*25], c='k', lw=0.4)
            axes[0,0].plot([15e2, 15e3], [1.5e3, 1.5e4], c='k', lw=0.4)
            axes[0,0].plot([2.5e3, 2.5e4], [5, 5*10**0.8], c='k', lw=0.4)
        elif mm == 'K' and 'unsym' in cn : 
            axes[0,0].plot([0.03,0.03*5], [0.04,0.04*25], c='k', lw=0.4)
            axes[0,0].plot([2e3, 2e4], [1.5e3, 1.5e4], c='k', lw=0.4)
            axes[0,0].plot([2.5e3, 2.5e4], [7.5, 7.5*10**0.9], c='k', lw=0.4)
        elif mm == 'Ag' and '120' in cn :
            axes[0,0].plot([0.03,0.03*5], [0.03,0.03*25], c='k', lw=0.4)
            axes[0,0].plot([3e3, 3e4], [2e3, 2e4], c='k', lw=0.4)
            axes[0,0].plot([3.5e3, 3.5e4], [6, 6*10**0.75], c='k', lw=0.4)
        elif mm == 'Ag' and 'unsym' in cn : 
            axes[0,0].plot([0.03,0.03*5], [0.02,0.02*25], c='k', lw=0.4)
            axes[0,0].plot([2.5e3, 2.5e4], [1.5e3, 1.5e4], c='k', lw=0.4)
            # axes[0,0].plot([2.5e3, 2.5e4], [5, 5*10**0.9], c='k', lw=0.4)
            axes[0,0].plot([4e3, 4e4], [1.5, 1.5*10**0.75], c='k', lw=0.4)
        elif mm == 'K' and '120' in cn : 
            axes[0,0].plot([0.03,0.03*5], [0.06,0.06*25], c='k', lw=0.4)
            axes[0,0].plot([2e3, 4e4], [300, 300*20**0.9], c='k', lw=0.4)
        axes[0,0].plot([0.04,10], [31.36, 31.36], c='k', lw=0.4, ls='--')
    
    # ===== (b) exponent vs distance =====
    fit_points = 21
    p0 = [1, 0]
    
    exp_alpha = np.array([cf(linfit, np.log10(a2.index.values[x:x+fit_points]), 
                             np.log10(a2.r2.values[x:x+fit_points]),p0)[0][0] for x in range(10,len(a2)-fit_points)])
    exp_times = a2.index.values[10+fit_points//2:-fit_points//2]
    exp_rs = np.sqrt(a2.r2.values[10+fit_points//2:-fit_points//2])
    # axes[0,1].plot(exp_times[exp_times >=0.8], exp_alpha[exp_times >=0.8], label=label, c=f'C{i}')
    axes[0,1].plot(exp_rs[exp_times >=0.8], exp_alpha[exp_times >=0.8], label=label, c=cc)
        
    ## always plot short
    try :
        a2s = a2s.loc[:0.8]
        exp_alpha = np.array([cf(linfit, np.log10(a2s.index.values[x:x+fit_points]), 
                                 np.log10(a2s.r2.values[x:x+fit_points]),p0)[0][0] for x in range(1,len(a2s)-fit_points)])
        exp_times = a2s.index.values[1+fit_points//2:-fit_points//2]
        exp_rs = np.sqrt(a2s.r2.values[1+fit_points//2:-fit_points//2])
        # axes[0,1].plot(exp_times, exp_alpha, c=f'C{i}', ls='--')
        axes[0,1].plot(exp_rs[exp_times <=0.8], exp_alpha[exp_times <=0.8], c=cc, ls='--')
    except: pass
    print(f'computed the exponent of MSD vs time for {mm} {cn} {T1}.')
    
    axes[0,1].legend(title=leg_title, loc='lower right')
    if guides: 
        axes[0,1].plot([0.03,3e5],[1,1], c='grey', lw=0.4, ls='--')
        axes[0,1].plot([5.6,5.6],[0,2.1], c='grey', lw=0.4, ls='--')
    
    # ===== (c) dx prep, and dx raw =====
    if T1 == T1_dx: 
        ## try loading a pre-computed dx file
        dx_glob = glob.glob(plane.folder+f'/{mm}-*-dx-{T1}K*ps.csv')
        dx = None
        try: 
            dx = pd.read_csv(dx_glob[0])
            dx = dx.set_index(['dx','time']).unstack().apply(lambda col: col/col.sum(), axis=0)
            dx.columns = [x[1] for x in dx.columns]
        except:
            print(f'could not load a dx file for {mm} {cn} {T1}K')
            continue
        
        ## apply binning by time intervals
        time_tuples = [ (round(x*0.8), round(x*1.2)) for x in dx_times]
        time_intervals = pd.IntervalIndex.from_tuples(time_tuples)
        time_spectra = dx.T.groupby(pd.cut(dx.T.index,time_intervals)).agg('mean').T
        
        ## normalize each column to sum to 1
        time_spectra = time_spectra / time_spectra.sum() / (time_spectra.index[1]-time_spectra.index[0])
        ## and rename the columns as something legible
        col_names = [f'{x[0]}-{x[1]} ps' if max(x) < 1000 else f'{int(x[0])//1000}-{int(x[1])//1000} ns' for x in time_tuples]
        time_spectra.rename(columns = dict(zip(time_spectra.columns,col_names)), inplace=True)
        
        ## plot each column
        for col in time_spectra.columns :
            xvals = time_spectra.loc[time_spectra[col] != 0].index
            axes[1,0].plot(xvals, time_spectra.loc[time_spectra[col] != 0, col], 
                    label=col, c=next(dx_colors))
        
        axes[1,0].legend(title=leg_title + f' {T1}K')
        
    # ===== (d) dx binned by hops and rescaled =====
    if T1 == T1_dx : 
        ## find the variances in dx to later rescale by them
        col_sigmas = list()
        for col, t in zip(time_spectra.columns, dx_times):
            col_variance = time_spectra[col] * time_spectra.index.values **2 / time_spectra[col].sum()
            col_sigma = np.sqrt(col_variance.sum())
            print(f'{mm} {cn} {T1}K : {t} ps, sigma = {col_sigma:.2f} AA')
            col_sigmas.append(col_sigma)
            
        ## numbers of hops from the Cartesian displacements
        x_bins = (np.unique(dx.index.values // hop_length) * 2 - 1 ) * (hop_length / 2)
        x_bins = np.insert(np.append(x_bins,max(x_bins)+hop_length), 0, min(x_bins)-hop_length)
        
        ## apply binning by number of hops
        time_spectra = time_spectra.groupby(pd.cut(time_spectra.index,x_bins)).agg('sum')
        time_spectra.index = (x_bins[:-1] + x_bins[1:])/2
        
        ## normalize each column to sum to 1
        time_spectra = time_spectra / time_spectra.sum() / (time_spectra.index[1]-time_spectra.index[0])
        
        ## plot each column
        for col, sigma in zip(time_spectra.columns, col_sigmas) :
            xvals = time_spectra.loc[time_spectra[col] != 0].index
            axes[1,1].plot(xvals/sigma, 
                    time_spectra.loc[time_spectra[col] != 0, col]*sigma, 
                    label=col, c=next(dx_colors))
        
        axes[1,1].legend(title=leg_title + f' {T1}K')
        
        ## plot a Laplacian & a Gaussian as benchmarks
        sigmas = np.linspace(-10, 10, 101)
        gauss = np.exp(-sigmas**2/2) / sum(np.exp(-sigmas**2/2)) * len(sigmas)/(max(sigmas)-min(sigmas))
        axes[1,1].plot(sigmas, gauss, c='grey', ls=':')
        laplace = np.exp(-abs(sigmas)*np.sqrt(2)) / sum(np.exp(-abs(sigmas)*np.sqrt(2))) * len(sigmas)/(max(sigmas)-min(sigmas))
        axes[1,1].plot(sigmas, laplace, c='k', ls=':')
    
    # ===== (e) Gs fitting: hardcoded time bounds =====
    if T1 in T1s_gs :
        
        ## try loading a pre-computed Gself file
        gs = hu.load_gs(plane.folder+f'/{mm}-*-gs-{T1}K*ps.csv', 'fitting')
        if gs is None: continue
    
        for h, (rs, ls) in enumerate(zip(rs_list, ['--', '-', '-.', ':'])) :
            s = gs.loc[min(rs):max(rs),0.1:5e4].sum().reset_index()
            s.columns = ['time','gs']
            s.set_index('time', inplace=True)
            # s = s/s.max()
            l = f'{label}, <{h+1} hop{"s" if h != 0 else ""}'
            axes[2,0].plot(s, label = l, c=cc, ls=ls)
            
            # ## fit and plot the fit
            # try: 
            #     s = s.loc[1:]
            #     popt, perr = expectation_multi_method(s, method, aggregated=True, verbose=True)
            #     if method == 'simple' : ax.plot(s.index.values, exp_decay(s.index.values, *popt), c='k', ls=':')
            #     elif method == 'stretch' : ax.plot(s.index.values, kww_decay_break(s.index.values, *popt), c='k', ls=':')
            #     print(f'fitting {mm} {cn} {T1} {min(rs):.1f}-{max(rs):.1f}AA : {popt[1]:.1f}±{perr[1]:.1f} ps, beta={1.00 if len(popt)<4 else popt[3]:.2f}, tstar={0 if len(popt)<4 else popt[4]:.2f}')
            # except : pass
            
            ## inverse interpolating function to plot the 1/e time
            int_fun = interp1d(s.gs.values, s.index.values)
            try : axes[2,0].plot(int_fun(1/np.e), 1/np.e, marker='o', ls='', fillstyle='full',
                                    mfc='yellow', mec='k', zorder=3, markersize=4)
            except : print(f'for {mm} {cn} {T1}, not all radii decay to 1/e')
        
        axes[2,0].legend(title=leg_title)
        if guides: axes[2,0].plot([1e3,3e4],[1/np.e,1/np.e], c='grey', lw=0.4, ls='--')
    
    # ===== (f) C_D(t) =====
    
    if T1 not in cd_exclude :
    
        start = dt.now()
        svals = np.logspace(-5, 2, 4000) # if not short else np.logspace(-6,5,3000)
        
        ## Laplace transform of C_D(t)
        cds = hu.fluctuation_kernel(a2, svals, dim=dim)
        
        try: cdt = hu.stehfest_inverse(cds, a2.index.values[1:-1])
        except :
            print(f'could not append inverse transform for {mm} {cn} {T1}')
            break
        cdt = pd.DataFrame({'time':a2.index.values[1:-1],'cdt':cdt}).set_index('time')
        
        if na_bdp_unsym : cdt = cdt.where(cdt < cd_break, cd_transform)
        
        axes[2,1].plot(cdt.cdt.loc[0.2:a2.index.max()/3+1], label=label, c=cc) 
        
        ## create the interpolator for plotting little stars based on Gs
        ## try loading a pre-computed Gself file
        gs = hu.load_gs(plane.folder+f'/{mm}-*-gs-{T1}K*ps.csv', 'cdt', radii=rs_list)
        int_fun = interp1d(cdt.index.values, cdt.cdt)
        
        try: axes[2,1].scatter(gs, int_fun(gs), marker='o', facecolors='yellow', edgecolors='k', zorder=3, s=16)
        except : print('something wrong with plotting Gs * for {mm} {cn} {T1}')
        
        ## plot short-time separately
        cds_s = hu.fluctuation_kernel(a2s, np.logspace(0,4,1000), dim=dim)
        cdt_s = hu.stehfest_inverse(cds_s, a2s.index.values[1:-1])
        cdt_s = pd.DataFrame({'time':a2s.index.values[1:-1],'cdt':cdt_s}).set_index('time')
        axes[2,1].plot(cdt_s.cdt.loc[0.0085 if (mm == 'Na' and 'unsym' in cn) else 0.005:0.2], ls='--', c=cc) 
        
        print(f'done {T1}K, time taken {(dt.now()-start).total_seconds():.2f}')
        axes[2,1].plot([1e-3, 5e4], [0,0], c='grey', ls=':', lw=0.4)
        
        axes[2,1].legend(title=leg_title, loc='upper left')
    else : 
        print(f'skipping C_D(t) for {mm} {cn} {T1}')
        
## plot prettymaking
axes[0,0].set(xlim=[0.025,5e4], xscale='log', ylim=[1e-2,3e4], yscale='log',
              xlabel=r'Time lag $t$, ps', ylabel=r'$\langle \overline{r^2(t)} \rangle$, $\AA^2$')
axes[0,1].set(xlim=[0.4,30], xscale='log', ylim=[0,1.05], xlabel=r'$\langle \overline{ r(t) }\rangle,~\AA$',
              yticks=[0,0.2,0.4,0.6,0.8,1.], yticklabels=['0.0','0.2','0.4','0.6','0.8','1.0'],
              xticks=[1,10], xticklabels=['1','10'],
              ylabel=r'Exponent of $\langle \overline{ r^2(t) }\rangle$')
axes[1,0].set(xlim=[-28,28], ylim=[3e-5,None], yscale='log',
              xlabel=r'$\Delta x$, $\AA$', ylabel=r'$P(\Delta x)$, $\AA^{-1}$')
axes[1,1].set(xlim=[-7,7], ylim=[1e-5,None], yscale='log',
              xlabel=r'$(\Delta x)/\sigma_{\Delta x}$', ylabel=r'$P(\Delta x)$, $\sigma_{\Delta x}^{-1}$')
# axes[2,0].set(ylim=[0,13.5], xlim=[0.5,9e3], xscale='log',
#               xlabel=r'Time lag $t$, ps', ylabel=r'Distance $r,~\AA$')
axes[2,0].set(xlim=[0.1,5e4], xscale='log', ylim=[0,1.04],
              ylabel=r'$G_s~r^2$, a.u.', xlabel=r'Time lag $t$, ps')
axes[2,1].set(xlim=[5e-3,5e3], xscale='log',
              xlabel=r'Time lag $t$, ps', ylabel=r'$C_D(t)$')

## create the broken axis
if na_bdp_unsym :
    r1 = Rectangle((4.5e-3,cd_break),5.2e3,cd_break_top-cd_break, lw=0, 
                   facecolor='w', clip_on=False, transform=axes[2,1].transData, zorder=3)
    axes[2,1].add_patch(r1)
    
    kwargs = dict(transform=axes[2,1].transData, color='k', clip_on=False, lw=0.75,zorder=4)
    axes[2,1].plot(5e-3*np.array([10**-0.05,10**0.05]), [cd_break-0.05,cd_break+0.05],**kwargs)
    axes[2,1].plot(5e-3*np.array([10**-0.05,10**0.05]), [cd_break_top-0.05,cd_break_top+0.05],**kwargs)
    axes[2,1].plot(5e3*np.array([10**-0.05,10**0.05]), [cd_break-0.05,cd_break+0.05],**kwargs)
    axes[2,1].plot(5e3*np.array([10**-0.05,10**0.05]), [cd_break_top-0.05,cd_break_top+0.05],**kwargs)
    # axes[2,1].set(yticks=[0,2,4,cd_transform(10), cd_transform(15)], yticklabels=['0','2','4','10','15'])
    axes[2,1].set(yticks=[0,2,4,cd_transform(6), cd_transform(8)], yticklabels=['0','2','4','6','8'])
    axes[2,1].set(ylim=[-0.7,7.1])

fig.tight_layout(pad=0.5, h_pad=0.25)

# =============================================================================
# %% Figure 3 top row: spectra of conductivity
# =============================================================================

## Parameters: 
    
start_1 = 0
start_step = 10       ## [ps] interval for sampling eMSD
durations = np.round(np.logspace(0.4,3.4),2)     ## [ps] 2.5-2500 ps, 50 pts

enforce_indep = True ## make start_step >= duration, for all durations

# rs_list = [[0.01,1.7],[0.01, 4.6]] ## to plot two-hop relaxation as the Jonscher cutoff
rs_list = []

# ========== automatic things below this line ==========

## three-panel top row
fig, axes = plt.subplots(1,3, sharey=True, figsize=(12,4))
# fig, axes = plt.subplots(3,1, sharex=True, figsize=(4,9))

# ===== (a) Na-beta-doubleprime spectra vs T1 =====

## load three planes
planes_to_load = all_planes.query('metal == "Na" & config == "unsym_0" & T1 in [230,300,473]')
macro_planes_data = hu.load_macro_planes(planes_to_load).sort_values(by='T1')
colors = cycle([batlow_even(j) for j in np.linspace(0, 1, len(macro_planes_data))])
ax = axes[0]

## load and plot Na doubleprime conductivity spectra
for i, plane in macro_planes_data.iterrows():
                
    ph = plane.phase; mm = plane.metal; cn = plane.config
    st = plane.stoich; ex = plane.exclude
    T1 = plane.T1; folder = plane.folder
    
    N = len(plane.atoms.query('atom == @mm'))
    cell = plane.cell
    
    dcoms = list()
    
    ## load a pre-corrected CoM trajectory
    cor = False
    try:
        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
            com = pd.read_csv(cor_fname[0]).set_index('time')
            print(f'\nLoaded a corrected CoM trajectory for {mm} {cn} T1={T1}K')
            cor = True
    except : 
        com = None; continue
    
    dtt = com.index[1]-com.index[0]
                
    ## average multiple starts
    for duration in durations: 
        
        ## enforce start_step >= duration 
        if enforce_indep and start_step < duration : start_step = duration
        
        if com.index.max() <= duration*4 :
            dcoms.append(np.nan)
            continue
    
        dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
        # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
        dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
        
        all_dcom2 = dr.dcom.values * N / 4
        
        dcoms.append(np.mean(all_dcom2))
        
    ## transform D_CoM to conductivity and plot it
    sigmas = sigma_T(N,cell,np.array(dcoms)*unit_conv)/T1
    this_marker = next(markers)
    ax.plot(1e12/durations, sigmas, this_marker+next(lines), label=f'eMSD, {T1}K', 
            markersize=5, c=next(colors))
    
    ## plot the two-hop relaxation time
    int_fun = interp1d(1e12/durations, sigmas, fill_value=1e-10, bounds_error=False)
    gs = np.array(hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=rs_list))
    ax.plot(1e12/gs, int_fun(1e12/gs), marker=this_marker, mfc='yellow', mec='k', zorder=3, ls='', fillstyle='full')
    
## plot literature values:
refs = {'Funke2007':52, 'Almond1984':32, 'Hoppe1991':51, 'Barker1976':44,
        'Kamishima2014':30, 'Kamishima2015':31}
    
## Funke & Banhatti (2007) - 473K
lit6 = pd.read_csv('./production/bdp-Na/Na_unsym_Funke2007_473K_lit.csv', names=['logfreq','sigma'])
axes[0].plot(10**lit6.logfreq, (10**lit6.sigma)/473, marker='o', mec='k', ls='', zorder=0, 
                mfc='none', markersize=4, label=f'Ref. {refs["Funke2007"]}, 473K') ## Funke $\\it{et\ al.}$ (2007), ($Li_{Al}^{\prime\prime}$)
## Hoppe & Funke (1991) - 220K
lit2 = pd.read_csv('./production/bdp-Na/Na_unsym_Hoppe1991_220K_lit.csv', names=['logfreq','sigma'])
axes[0].plot(10**lit2.logfreq, lit2.sigma, marker='o', mec='k', ls='', zorder=0,
                mfc='none', markersize=4, label=f'Ref. {refs["Hoppe1991"]}, 220K') ## Hoppe $\\it{et\ al.}$ (1991), ($Li_{Al}^{\prime\prime}$)
## Hoppe & Funke (1991) - 298K
lit5 = pd.read_csv('./production/bdp-Na/Na_unsym_Hoppe1991_298K_lit.csv', names=['logfreq','sigma'])
axes[0].plot(10**lit5.logfreq, lit5.sigma, marker='^', mec='k', ls='', zorder=0,
                mfc='none', markersize=4, label=f'Ref. {refs["Hoppe1991"]}, 298K') ## Hoppe $\\it{et\ al.}$ (1991) ($Li_{Al}^{\prime\prime}$)

## Almond et al (1984) - 237K
lit1 = pd.read_csv('./production/bdp-Na/Na_unsym_Almond1984_237K_lit.csv', names=['logfreq','sigma'])
axes[0].plot(10**lit1.logfreq, lit1.sigma, marker='d', mec='k', ls='', zorder=0,
                mfc='k', fillstyle='full', markersize=4, label=f'Ref. {refs["Almond1984"]}, 237K') ## Almond $\\it{et\ al.}$ (1984)
## Almond (1984) - 296K
lit3 = pd.read_csv('./production/bdp-Na/Na_unsym_Almond1984_296K_lit.csv', names=['logfreq','sigma'])
axes[0].plot(10**lit3.logfreq, lit3.sigma, marker='s', mec='k', ls='', zorder=0, 
                mfc='k', fillstyle='full', markersize=4, label=f'Ref. {refs["Almond1984"]}, 296K') ## Almond $\\it{et\ al.}$ (1984)

## make plot pretty
axes[0].set(xlim=[8e5,6e11], ylim=[6e-4,1.2], xscale='log', yscale='log',
            xlabel=r'$\nu=1/t$, Hz', ylabel=r'$\sigma_{2D}(t;\Delta)$, S/cm')
axes[0].legend(title=r'Na $\beta^{\prime\prime}$', ncol=2, handletextpad=0.5, 
               handlelength=1.5, columnspacing=0.4, loc='upper left')

## add guidelines
axes[0].plot([1e6, 2e9], [6e-3, 6e-3*2000**0.15], c='grey', lw=0.4)
axes[0].plot([1e6, 2e8], [1e-3, 1e-3*200**0.15], c='grey', lw=0.4)
axes[0].plot([4e9, 4e11], [0.06*100**-0.7, 0.06], c='grey', lw=0.4)

# ===== (b) Na-beta spectra 300K =====

## load planes
planes_to_load = all_planes.query('metal == "Na" & config == "120_4" & T1 == 300')
macro_planes_data = hu.load_macro_planes(planes_to_load)
ax = axes[1]

## load and plot Na beta conductivity spectra
for i, plane in macro_planes_data.iterrows():
                
    ph = plane.phase; mm = plane.metal; cn = plane.config
    st = plane.stoich; ex = plane.exclude
    T1 = plane.T1; folder = plane.folder
    
    N = len(plane.atoms.query('atom == @mm'))
    cell = plane.cell
    
    dcoms = list()
    
    ## load a pre-corrected CoM trajectory
    cor = False
    try:
        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
            com = pd.read_csv(cor_fname[0]).set_index('time')
            print(f'\nLoaded a corrected CoM trajectory for {mm} {cn} T1={T1}K')
            cor = True
    except : 
        com = None; continue
    
    dtt = com.index[1]-com.index[0]
                
    ## average multiple starts
    for duration in durations: 
        
        ## enforce start_step >= duration 
        if enforce_indep and start_step < duration : start_step = duration
        
        if com.index.max() <= duration*4 :
            dcoms.append(np.nan)
            continue
    
        dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
        # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
        dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
        
        all_dcom2 = dr.dcom.values * N / 4
        
        dcoms.append(np.mean(all_dcom2))
        
    sigmas = sigma_T(N,cell,np.array(dcoms)*unit_conv)/T1
    this_marker = next(markers)
    ax.plot(1e12/durations, sigmas, this_marker+next(lines), label=f'eMSD, {T1}K', 
            markersize=5, c=batlow_even(0))
    
    ## plot the two-hop relaxation time
    int_fun = interp1d(1e12/durations, sigmas, fill_value=1e-10, bounds_error=False)
    gs = np.array(hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=rs_list))
    ax.plot(1e12/gs, int_fun(1e12/gs), marker=this_marker, mfc='yellow', mec='k', zorder=3, ls='', fillstyle='full')
    
## plot literature values

## 2 literature datasets, Barker (1976) - 300K ## Barker $\\it{et\ al.}$ (1976)
lit1 = pd.read_csv('./production/beta-Na/Na_120_Barker1976_flux_lit.csv', names=['logfreq','sigma'])
axes[1].plot(10**lit1.logfreq, lit1.sigma, marker='D', mec='k', markersize=4, ls='', zorder=0,
           mfc='none', label=f'Ref. {refs["Barker1976"]}, 300K, flux')
lit2 = pd.read_csv('./production/beta-Na/Na_120_Barker1976_melt_lit.csv', names=['logfreq','sigma'])
axes[1].plot(10**lit2.logfreq, lit2.sigma, marker='s',mec='k', markersize=4, ls='', zorder=0,
           mfc='none', label=f'Ref. {refs["Barker1976"]}, 300K, melt') 

## Kamishima (2015) - 300K # Kamishima $\\it{et\ al.}$ (2015)
axes[1].plot(1e7, 0.011692, marker='>', mec='k', markersize=4, ls='', zorder=0,
           mfc='k', fillstyle='full', label=f'Ref. {refs["Kamishima2015"]}, 300K')

## make plot pretty
axes[1].set(xlim=[7e6,6e11], xscale='log', xlabel=r'$\nu=1/t$, Hz')
axes[1].legend(title=r'Na $\beta$', handletextpad=0.5, handlelength=1.5)

## add guidelines
axes[1].plot([1e7, 4e8], [6e-3, 6e-3*40**0.1], c='grey', lw=0.4)
axes[1].plot([4e9, 4e11], [0.05*100**-0.6, 0.05], c='grey', lw=0.4)

# ===== (c) Ag-beta spectra 300K =====

## load planes
planes_to_load = all_planes.query('metal == "Ag" & config == "120_4" & T1 == 300')
macro_planes_data = hu.load_macro_planes(planes_to_load)
ax = axes[2]

## load and plot Ag beta conductivity spectra
for i, plane in macro_planes_data.iterrows():
                
    ph = plane.phase; mm = plane.metal; cn = plane.config
    st = plane.stoich; ex = plane.exclude
    T1 = plane.T1; folder = plane.folder
    
    N = len(plane.atoms.query('atom == @mm'))
    cell = plane.cell
    
    dcoms = list()
    
    ## load a pre-corrected CoM trajectory
    cor = False
    try:
        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
            com = pd.read_csv(cor_fname[0]).set_index('time')
            print(f'\nLoaded a corrected CoM trajectory for {mm} {cn} T1={T1}K')
            cor = True
    except : 
        com = None; continue
    
    dtt = com.index[1]-com.index[0]
                
    ## average multiple starts
    for duration in durations: 
        
        ## enforce start_step >= duration 
        if enforce_indep and start_step < duration : start_step = duration
        
        if com.index.max() <= duration*4 :
            dcoms.append(np.nan)
            continue
    
        dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
        # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
        dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
        
        all_dcom2 = dr.dcom.values * N / 4
        
        dcoms.append(np.mean(all_dcom2))
        
    sigmas = sigma_T(N,cell,np.array(dcoms)*unit_conv)/T1
    this_marker = next(markers)
    ax.plot(1e12/durations, sigmas, this_marker+next(lines), label=f'eMSD, {T1}K', markersize=5,
            c=batlow_even(0))
    
    ## plot the two-hop relaxation time
    int_fun = interp1d(1e12/durations, sigmas, fill_value=1e-10, bounds_error=False)
    gs = np.array(hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=rs_list))
    ax.plot(1e12/gs, int_fun(1e12/gs), marker=this_marker, mfc='yellow', mec='k', zorder=3, ls='', fillstyle='full')
    
## plot literature values

## Barker (1976) melt - 300K # Barker $\\it{et\ al.}$ (1976) melt
lit21 = pd.read_csv('./production/beta-Ag/Ag_120_Barker1976_melt_lit.csv', names=['logfreq','sigma'])
axes[2].plot(10**lit21.logfreq, lit21.sigma, marker='s', mec='k', markersize=4, ls='',
           mfc='none', label=f'Ref. {refs["Barker1976"]}, 300K, melt', zorder=0)

## 3 samples from Kamishima (2014) - near 300K ## Kamishima $\\it{et\ al.}$ (2014)
lit22 = pd.read_csv('./production/beta-Ag/Ag_120_Kamishima2014_296K_S1_lit.csv', names=['logfreq','sigma'])
axes[2].plot(10**lit22.logfreq, lit22.sigma, marker='o', mec='k', markersize=4, ls='',
           mfc='k', fillstyle='full', label=f'Ref. {refs["Kamishima2014"]}, 296K, A', zorder=0)
lit23 = pd.read_csv('./production/beta-Ag/Ag_120_Kamishima2014_286K_S2_lit.csv', names=['logfreq','sigma'])
axes[2].plot(10**lit23.logfreq, lit23.sigma, marker='^', mec='k', markersize=4, ls='',
           mfc='k', fillstyle='full', label=f'Ref. {refs["Kamishima2014"]}, 286K, B', zorder=0)
lit24 = pd.read_csv('./production/beta-Ag/Ag_120_Kamishima2014_299K_S3_lit.csv', names=['logfreq','sigma'])
axes[2].plot(10**lit24.logfreq, lit24.sigma, marker='v', mec='k', markersize=4, ls='',
           mfc='k', fillstyle='full', label=f'Ref. {refs["Kamishima2014"]}, 299K, C', zorder=0)


## make plot pretty
axes[2].set(xlim=[5e5,6e11], xscale='log', xlabel=r'$\nu=1/t$, Hz')
axes[2].legend(title=r'Ag $\beta$', handletextpad=0.5, handlelength=1.5)

## add guidelines
axes[2].plot([4e9, 4e11], [0.05*100**-0.6, 0.05], c='grey', lw=0.4)
axes[2].plot([2e6, 2e8], [3e-3, 3e-3*100**0.1], c='grey', lw=0.4)

fig.tight_layout(pad=0.5, w_pad=0.25)

# =============================================================================
# %% Fig 3 bottom row: Arrhenius plots
# =============================================================================

# Parameters:

start_1 = 0         ## [ps] time at which to start sampling CoM MSD
start_step = 2500    ## [ps] interval for sampling CoM MSD
start_last = 97500    ## [ps] last time at which to sample CoM MSD
  
duration = 2500     ## [ps] how long each sampling is

refs_dict = {'Davies(1986)':38, 'Briant(1980)':34, 'Bates(1981)':35,
             'Whittingham(1972)':55, 'Almond(1984)':32}

beta_refs_dict = {'Ag':53, 'K':55, 'Na':54}

# ========== automatic things below this line ==========

## array for multiple starts and its tuple description for lookup of pre-dones
starts = np.arange(start_1,start_last,start_step,dtype=float) 
spec=(start_1,start_step,start_last,duration)  

## pre-load and filter for the same computation conditions right away
sigmas_msd = pd.read_csv('./production/sigmas_msd.csv')
sigmas_msd.spec = sigmas_msd.spec.apply(eval)   
sigmas_msd_spec = sigmas_msd.query('spec == @spec')

# figure
fig, axes = plt.subplots(1,3, sharey=True, figsize=(12,4))

# ===== (d) Na-doubleprime: normal + quenched =====

planes_to_load = all_planes.query('metal == "Na" & config in ["unsym_0", "symm_1"] & T1 in [230,300,473,600]')
macro_planes_data = hu.load_macro_planes(planes_to_load)

## structure for new computations
new_sigmas_msd = list()

for i, plane in macro_planes_data.iterrows():
                
    all_dcom = list()
    all_derr = list()
    ph = plane.phase
    mm = plane.metal
    cn = plane.config
    st = plane.stoich
    ex = plane.exclude
    T1 = plane.T1
    com = plane.com
    N = len(plane.atoms.query('atom == @mm'))
    
    ## check this configuration was already computed
    pre_done = sigmas_msd_spec.query('metal == @mm & stoich == @st & exclude == @ex & T1 == @T1')
    
    assert len(pre_done) > 0, 're-compute {mm} {cn} {T1}'
    new_sigmas_msd.append(pre_done.to_dict('records')[0])
    # print(f'pre-done {variable}={var}, {variable2}={var2} T1={T1}K, sigma*T = {pre_done.sigt.iloc[0]:.2e} S*K/cm')
    
## convert all fitted sigmas to dataframe
new_sigmas_msd = pd.DataFrame(new_sigmas_msd)

## plot normal Na beta-doubleprime
sigts = new_sigmas_msd.query('config == "unsym_0"')

axes[0].errorbar(x=1000./sigts.T1.values, y=sigts.sigt, zorder=2.5, 
                 yerr=[sigts.sigt-sigts.sigt_20, sigts.sigt_80-sigts.sigt],
                 label=r'Na $\beta^{\prime\prime}$, eMSD', mfc=hu.metal_colors['Na'],
                 mec=hu.metal_colors['Na'], c=hu.metal_colors['Na'],
                 fillstyle='full', marker='o', ls='')

## plot literature values
lit_folder = macro_planes_data.lit_folder.unique()[0]
d_lit_files = sorted(glob.glob(lit_folder + f'/{mm}*sigma*lit.csv'), reverse=True)

for f, sym in zip(d_lit_files, ['o','s','D','v','^','<','>']) :
    d_lit = np.loadtxt(f, delimiter = ',')
    # print('loaded', f)

    ## find the author+year if they are in the filename
    auth = [x[0] for x in [re.findall('[A-z]*\(19\d\d\)$', x) for x in re.split('/|-|_| ',f)] if len(x) > 0][0]
    try: t_synth = [x[0] for x in [re.findall('1\d\d0$', x) for x in re.split('/|-|_| ',f)] if len(x) > 0][0]
    except: t_synth = None
    ref = refs_dict[auth]
    label = f'Ref. {ref}'
    if 'symm' in f : 
        label += ', quench'
        continue ## skip this for simplifying the plot
    elif t_synth is not None : 
        # label += f', {t_synth}C'
        if int(t_synth) == 1700 : continue

    ## scatterplot, can be updated to include errorbars
    axes[0].plot(d_lit[:,0], 10**d_lit[:,1], label=label, zorder=0, 
                mec='k', # if variable2 != 'metal' else hu.metal_colors[var],
                mfc=(0,0,0,0), marker=sym, linestyle='', markersize=5)
    
axes[0].set(ylabel='$\sigma$T [$\Omega^{{-1}}$ cm$^{{-1}}$ K]', xlabel='1000/T, K$^{-1}$',
            ylim=[1.5e-2,1.7e3], yscale='log', xlim=[1.3,4.45])
axes[0].legend(loc='lower left', title=r'Na $\beta^{\prime\prime}$', title_fontsize=10)

# ===== (e) K,Ag-doubleprime =====

planes_to_load = all_planes.query('metal in ["K", "Ag"] & config == "unsym_0" & T1 in [230,300,473,600]')
macro_planes_data = hu.load_macro_planes(planes_to_load)

## structure for new computations
new_sigmas_msd = list()

for i, plane in macro_planes_data.iterrows():
                
    all_dcom = list()
    all_derr = list()
    ph = plane.phase
    mm = plane.metal
    cn = plane.config
    st = plane.stoich
    ex = plane.exclude
    T1 = plane.T1
    com = plane.com
    N = len(plane.atoms.query('atom == @mm'))
    
    ## check this configuration was already computed
    pre_done = sigmas_msd_spec.query('metal == @mm & stoich == @st & exclude == @ex & T1 == @T1')
    
    assert len(pre_done) > 0, 're-compute {mm} {cn} {T1}'
    new_sigmas_msd.append(pre_done.to_dict('records')[0])
    # print(f'pre-done {variable}={var}, {variable2}={var2} T1={T1}K, sigma*T = {pre_done.sigt.iloc[0]:.2e} S*K/cm')

    
## convert all fitted sigmas to dataframe
new_sigmas_msd = pd.DataFrame(new_sigmas_msd)

## plot K beta-doubleprime
sigts = new_sigmas_msd.query('metal == "K"')

axes[1].errorbar(x=1000./sigts.T1.values, y=sigts.sigt, zorder=2.5, 
                  yerr=[sigts.sigt-sigts.sigt_20, sigts.sigt_80-sigts.sigt],
                  label=r'K $\beta^{\prime\prime}$, eMSD', c=hu.metal_colors['K'], 
                  mfc=hu.metal_colors['K'], mec=hu.metal_colors['K'],
                  fillstyle='full', marker='o', ls='')

## plot K beta-doubleprime
sigts = new_sigmas_msd.query('metal == "Ag"')

axes[1].errorbar(x=1000./sigts.T1.values, y=sigts.sigt, zorder=2.5, 
                  yerr=[sigts.sigt-sigts.sigt_20, sigts.sigt_80-sigts.sigt],
                  label=r'Ag $\beta^{\prime\prime}$, eMSD', c=hu.metal_colors['Ag'], 
                  mfc=hu.metal_colors['Ag'], mec=hu.metal_colors['Ag'],
                  fillstyle='full', marker='s', ls='')

## plot literature values
lit_folders = macro_planes_data.lit_folder.unique()
d_lit_files = sorted(glob.glob(lit_folder + '/*sigma*lit.csv'), reverse=True)

for lf in lit_folders:
    mm = re.split('-|/| ',lf)[-1]
    d_lit_files = sorted(glob.glob(lf + '/*sigma*lit.csv'), reverse=True)
    for f, sym in zip(d_lit_files, ['o','s','D','v','^','<','>']) :
        d_lit = np.loadtxt(f, delimiter = ',')
        # print('loaded', f)
    
        ## find the author+year if they are in the filename
        auth = [x[0] for x in [re.findall('[A-z]*\(19\d\d\)$', x) for x in re.split('/|-|_| ',f)] if len(x) > 0][0]
        try: t_synth = [x[0] for x in [re.findall('1\d\d0$', x) for x in re.split('/|-|_| ',f)] if len(x) > 0][0]
        except: t_synth = None
        ref = refs_dict[auth]
        label = mm + r' $\beta^{\prime\prime}$, ' + f'Ref. {ref}'
        # if 'symm' in f : label += ', quench'
        # elif t_synth is not None : label += f', {t_synth}C'
    
        ## scatterplot, can be updated to include errorbars
        axes[1].plot(d_lit[:,0], 10**d_lit[:,1], label=label, zorder=0, 
                    mec='k', # if variable2 != 'metal' else hu.metal_colors[var],
                    mfc=(0,0,0,0), marker=next(markers), linestyle='', markersize=5)

axes[1].set(xlabel='1000/T, K$^{-1}$', xlim=[1.3,4.45])
axes[1].legend(loc='lower left', title_fontsize=10)

# ===== (f) Ag,K,Na beta, 120_4 =====

planes_to_load = all_planes.query('config == "120_4" & T1 in [300,600,1000]')
planes_to_load = planes_to_load.query('not (T1 == 300 & metal == "K")')
macro_planes_data = hu.load_macro_planes(planes_to_load)

## structure for new computations
new_sigmas_msd = list()

for i, plane in macro_planes_data.iterrows():
                
    all_dcom = list()
    all_derr = list()
    ph = plane.phase
    mm = plane.metal
    cn = plane.config
    st = plane.stoich
    ex = plane.exclude
    T1 = plane.T1
    com = plane.com
    N = len(plane.atoms.query('atom == @mm'))
    
    ## check this configuration was already computed
    pre_done = sigmas_msd_spec.query('metal == @mm & stoich == @st & exclude == @ex & T1 == @T1')
    
    assert len(pre_done) > 0, 're-compute {mm} {cn} {T1}'
    new_sigmas_msd.append(pre_done.to_dict('records')[0])
    # print(f'pre-done {variable}={var}, {variable2}={var2} T1={T1}K, sigma*T = {pre_done.sigt.iloc[0]:.2e} S*K/cm')

## convert all fitted sigmas to dataframe
new_sigmas_msd = pd.DataFrame(new_sigmas_msd)

for i, mm in enumerate(new_sigmas_msd.metal.unique()) :
    sigts = new_sigmas_msd.query('metal == @mm')
    
    axes[2].errorbar(x=1000./sigts.T1.values, y=sigts.sigt, zorder=2.5, 
                  yerr=[sigts.sigt-sigts.sigt_20, sigts.sigt_80-sigts.sigt],
                  label=mm + r', eMSD', c=hu.metal_colors[mm], mfc=hu.metal_colors[mm], # + f', $t=${duration/1000} ns'
                  fillstyle='full', marker=next(markers), ls='')
    
    ## plot literature values
    lit_folder = macro_planes_data.query('metal == @mm & config == "120_4"').lit_folder.unique()[0]
    f = sorted(glob.glob(lit_folder + '/*sigma*lit.csv'), reverse=True)[0]
    d_lit = np.loadtxt(f, delimiter = ',')
    # print('loaded', f)

    ref = beta_refs_dict[mm]
    label = f' Ref. {ref}, {mm}'

    ## scatterplot, can be updated to include errorbars
    axes[2].plot(d_lit[:,0], 10**d_lit[:,1], label=label, zorder=0, mec='k', # if variable2 != 'metal' else hu.metal_colors[var],
                mfc=(0,0,0,0), marker=next(markers), linestyle='', markersize=5)
    # print('plotted', f)
    
axes[2].set(xlabel='1000/T, K$^{-1}$', xlim=[0.9,3.49])
axes[2].legend(loc='lower left', title=r'$\beta$-aluminas', ncol=2, title_fontsize=10)

## final figure pretty-making
fig.tight_layout(pad=0.5, w_pad=0.25)

# =============================================================================
# %% SI Figure NN : non-Gaussian & EB parameters for all relevant simulations
# ## (Block 7, option='a2')
# =============================================================================

option = 'a2' ## implemented here: 'a2', 'eb'
do_fft = False ## implemented for 'a2', 'burnett'; requires regularly spaced data
dimension = 2  ## stuff within the conduction plane is 2D; super-short is 3D

plot_gs = True ## if True, add 1/e times from self van Hove function decay
guides = True  ## plot log-log guidelines for long-time regimes 

rs_list = [[0.01,1.75],[0.01, 4.2]] ## default radii for van Hove decay points
rs_list = [[0.01,1.7],[0.01, 4.6]] ## default radii for van Hove decay points

## variable by which to plot stuff: variable is columns, variable2 is rows
variable = 'metal' 
variable2 = 'config'

eb_lag = 20

# ========== automatic things below this line ==========

planes_to_load = all_planes.query('config in ["unsym_0", "symm_1", "120_4"] & T1 in [230,300,473,600,1000]')
planes_to_load = planes_to_load.query('T1 != 473 or phase != "beta"').sort_values(by='config',ascending=False)
macro_planes_data = hu.load_macro_planes(planes_to_load)

## values of the first (metal) and second (config) variables in the loaded data
var_values = sorted(macro_planes_data[variable].unique())
var2_values = sorted(macro_planes_data[variable2].unique(), reverse=True)

## adjust colormap
batlow_even = LinearSegmentedColormap.from_list('batlow_even', hu.batlow_lightness_scaled(0.4,0.6))

## figure to make an Arrhenius plot
fig, axes = plt.subplots(len(var2_values), len(var_values), sharex=True, sharey='row',
                         figsize=(3.5*len(var_values),4.8*len(var2_values)))

## loop over values of the variable(s)
for r, var2 in enumerate(var2_values) :
    
    # ## make y axes shared by row
    # for ax in axes[r,:]:
    #     axes[r,0]._shared_y_axes.join(ax,axes[r,0])
        
    for c, var in enumerate(var_values):
        
        ## set the current axes
        ax = axes[r,c]
        
        ## subset planes
        subset = macro_planes_data.query(f'{variable} == @var & {variable2} == @var2').sort_values(by="T1", ascending=True)
        guide_T1 = subset.T1.max()
        
        ## make a color map
        colors = [batlow_even(j) for j in np.linspace(0, 1, len(subset))]
        
        ## iterate through all data planes
        for i, (index, plane) in enumerate(subset.iterrows()):
            if plane[variable] == var and plane[variable2] == var2:
                
                mm = plane.metal; st = plane.stoich; cn = plane.config
                ex = plane.exclude; T1 = plane.T1; folder = plane.folder
                ph = plane.phase
                
                ## set a plotting label 
                # label = f'{mm}$_{{{int(st)/100:.2f}}}$ {T1}K'
                label = f'{T1}K'
                # label = str(var2) + units[variable2]
                leg_title = f'{mm} {phases[ph]}' + (' quench' if 'symm' in cn else '')
                
                ## interpolation for van Hove
                int_fun = None
                
                ## load the 2D file - leaving out the "split" files
                if dimension == 2 :
                    a2_xys = glob.glob(folder+f'/{mm}*{ex}-a2xy*{T1}K*ps.csv')
    
                    ## load the a2 file if exactly one exists, else complain
                    if a2_xys :
                        if len(a2_xys) > 1 : a2_xys = sorted(a2_xys, reverse=True,
                                        key = lambda x : eval(re.split('-|_| ',x)[-1][:-6]))
        
                        ## read the a2 file - options for csv file
                        a2 = pd.read_csv(a2_xys[0], sep=',').set_index('time')
                    else : print(f'could not load a 2D a2 file for plane {mm} {cn} {T1}K')
                else : a2 = plane.a2
                    
                ## recalculate a2 for the right number of dimensions
                a2.a2 = dimension * a2.r4 / a2.r2 ** 2 / (dimension+2) - 1
                
                ## non-Gaussian parameter
                if option == 'a2' and not do_fft :
                    ax.plot(a2.a2, label=label, c=colors[i])
                    
                    ## create the interpolator for plotting little stars based on Gs
                    if plot_gs : int_fun = interp1d(a2.index.values, a2.a2)
                
                    ## Plot points from van Hove function
                    if plot_gs and int_fun is not None :
                        try :
                            gs = hu.load_gs(plane.folder+f'/{mm}-*-gs-{T1}K*ps.csv', option, radii=rs_list)
                            ax.plot(gs, int_fun(gs), marker='o', mfc='yellow', ls='', markersize=5,
                                    mec='k', zorder=3, fillstyle='full')
                        except ValueError :
                            print(f'something wrong with Gs for {mm} {cn} {T1}, check fractional/real computation.')
                            
                    ax.legend(title=leg_title, loc='lower left')
                    
                    ## plot log-log guidelines
                    if guides and T1 == guide_T1 :
                        # axes[0].plot([0.25, 0.25*10**0.333],[3e-2,3e-1],c='k', lw=0.4)
                        if mm == 'K' and '120' in cn :
                            ax.plot([30, 3000],[1.5,1.5*10**-0.8],c='k', lw=0.4)
                        elif mm == 'Na' and 'unsym' in cn :
                            ax.plot([5,100], [0.4,0.4*20**-0.8], c='k', lw=0.4)
                            ax.plot([60,3e3], [2,2*50**-0.4], c='k', lw=0.4)
                        elif mm == 'Na' and '120' in cn :
                            ax.plot([8,400], [1,50**-0.5], c='k', lw=0.4)
                        elif mm == 'K' and 'unsym' in cn :
                            ax.plot([10,100], [0.35,0.35*10**-0.9], c='k', lw=0.4)
                        elif mm == 'Na' and 'symm' in cn :
                            ax.plot([10,100], [0.2,0.2*10**-0.8], c='k', lw=0.4)
                        elif mm == 'K' and 'symm' in cn :
                            ax.plot([7,70], [0.35,0.35*10**-0.9], c='k', lw=0.4)
                        elif mm == 'Ag' and 'symm' in cn :
                            ax.plot([10,100], [0.35,0.35*10**-0.7], c='k', lw=0.4)
                        elif mm == 'Ag' and '120' in cn :
                            ax.plot([15,150], [0.6,0.6*10**-0.6], c='k', lw=0.4)
                            ax.plot([50,2500], [1,50**-0.5], c='k', lw=0.4)
                            ax.plot([500,5e3], [2,2*10**-0.4], c='k', lw=0.4)
                        elif mm == 'Ag' and 'unsym' in cn :
                            ax.plot([20, 200], [0.3, 0.3*10**-0.8], c='k', lw=0.4)
                            ax.plot([4e2, 4e3], [2, 2*10**-0.4], c='k', lw=0.4)
                
                elif option == 'eb' :
                    
                    ## Load and plot EB
                    try:
                        eb_glob = glob.glob(plane.folder+f'/*eb*{T1}K*{int(eb_lag)}ps.csv')
                        eb = pd.read_csv(eb_glob[0]).set_index('time')
                        ax.plot(eb.eb, label=label, c=colors[i]) # + f', $t={eb_lag}$ ps'
                    except: 
                        print(f'could not load the first EB file for {mm} {cn} {T1}K: ')                
                    
                    ax.legend(title=leg_title, loc='lower left')
                    
                    ## plot log-log guidelines
                    if guides and T1 == guide_T1 : 
                        if mm == 'K' and '120' in cn:
                            ax.plot([1e4,8e4],[0.1, 0.1*8**-0.75], lw=0.4, c='k')
                        elif mm == 'Na' and 'unsym' in cn:
                            ax.plot([7e3, 7e4], [0.25,0.25*10**-0.3], c='k', lw=0.4)
                            ax.plot([1e3, 1e4], [0.015,0.015*10**-0.9], c='k', lw=0.4)
                        elif mm == 'Na' and '120' in cn:
                            ax.plot([7e3,7e4], [0.02,0.02*10**-1], c='k', lw=0.4)
                            ax.plot([12e3,72e3], [0.11,0.11*6**-0.6], c='k', lw=0.4)
                        elif mm == 'K' and 'unsym' in cn :
                            ax.plot([12e3,72e3], [0.015,0.015*6**-0.6], c='k', lw=0.4)
                            ax.plot([1e3,1e4], [0.016,0.016*10**-1], c='k', lw=0.4)
                        elif mm == 'Na' and 'symm' in cn :
                            ax.plot([12e3,72e3], [0.017,0.017*6**-0.6], c='k', lw=0.4)
                            ax.plot([1e3,1e4], [0.02,0.02*10**-1], c='k', lw=0.4)
                        elif mm == 'Ag' and '120' in cn :
                            ax.plot([2e4, 7e4], [0.04,0.04*3.5**-1], c='k', lw=0.4)
                            ax.plot([3e3, 14e3], [0.13, 0.13*(14/3)**-0.6], c='k', lw=0.4)
                            # axes[2].plot([3e3, 3e4], [0.3, 0.3*10**-0.3], c='k', lw=0.4)
                        elif mm == 'Ag' and 'unsym' in cn :
                            ax.plot([2e3, 2e4], [0.02, 0.02*10**-1], c='k', lw=0.4)
                        elif mm == 'Ag' and 'symm' in cn :
                            axes[1,0].plot([9e2,9e3], [0.045, 0.045*10**-0.9], c='k', lw=0.4)
                        elif mm == 'K' and 'symm' in cn :
                            ax.plot([9e2,9e3], [0.05, 0.005], c='k', lw=0.4)
                        else : pass
                    
                    
    
## make axes pretty
if option == 'a2' :
    # axes[0,0].set(xlim=[0.05,5e4], xscale='log', ylim=[0.02,None], yscale='log',yticks=[0.1,1.0,10])
    for ax in axes[:,0] : 
        ax.set(ylabel='Non-Gauss. Param.', yscale='log',ylim=[0.02,None])
        ax.set(yticks=[0.1,1.0,10] if max(ax.get_ylim()) > 10 else [0.1,1.0],
               yticklabels=['0.1','1.0','10'] if max(ax.get_ylim()) > 10 else ['0.1','1.0'])
    for ax in axes[:,1:].flat : ax.set(yticks=[])
    for ax in axes[-1,:] : ax.set(xlabel=r'Time lag $t$, ps', xlim=[0.05,5e4], xscale='log')
elif option == 'eb' : 
    for ax in axes[:,0] :
        ax.set(xlim=[5*eb_lag, 9e4], xscale='log', ylim=[1e-3, 3], yscale='log',
               ylabel=f'EB at $t=${eb_lag} ps',
               yticks=[0.01, 0.1, 1], yticklabels=['.01', '0.1', '1.0']
               )
    for ax in axes[-1,:] : ax.set(xlabel='Simulation Time $\Delta$, ps')

else : pass

fig.tight_layout(pad=0.5, w_pad=0.1)

# =============================================================================
# %% Extended Data Figure NN : Distributions of r^2_CoM
# ## top row: rescale = False, bottom row: rescale = True
# =============================================================================

option = 'hist'  ## 'spectra' for conductivity, 'hist' for distributions of DCOM, 
## variable by which to plot stuff
variable = 'T1' ## << pick a variable
start_1 = 0         ## [ps] time at which to start sampling CoM MSD
start_step = 10    ## [ps] interval for sampling CoM MSD  

durations = np.array([2.5,25,250,2500])     ## [ps] for histograms

rescale=True        ## divide the DCOM distribution by its stdev, with hist

enforce_random = False ## flag to enforce start_step >= duration 

# ========== automatic things below this line ==========

## load three planes
planes_to_load = all_planes.query('metal == "Na" & config == "unsym_0" & T1 in [230,300,473,600]')
macro_planes_data = hu.load_macro_planes(planes_to_load)

## values of the first variable in the loaded data
var_values = sorted(macro_planes_data[variable].unique())

## initialize a second variable
variable2 = None

## deduce a second variable
if len(macro_planes_data) > len(var_values) : ## a second variable is varied
    for rv in [x for x in relevant_vars if x != 'z']: 
        if rv in macro_planes_data.columns and len(set(macro_planes_data[rv])) > 1 and rv != variable: 
            variable2 = rv
            break
else: variable2 = 'config'
var2_values = sorted(macro_planes_data[variable2].unique())
        
## figure to plot the distributions
## do not share x-scale if each axes is at a different temperature ## sharex=(variable != 'T1'),
fig, axes = plt.subplots(1, len(var_values), sharey=True, sharex=rescale,
                         figsize=(3.2*len(var_values),3.75))
if len(var_values) < 2: axes = [axes]

## structure for new computations
new_sigmas_msd = list()

## loop over the values of the variables
for var, ax in zip(var_values, axes) :
    for var2 in var2_values:
        for i, plane in macro_planes_data.iterrows():
            if plane[variable] == var and plane[variable2] == var2:
                
                ph = plane.phase; mm = plane.metal; cn = plane.config
                st = plane.stoich; ex = plane.exclude
                T1 = plane.T1
                # com = plane.com
                N = len(plane.atoms.query('atom == @mm'))
                cell = plane.cell
                
                dcoms = list()
                
                ## load a pre-corrected result
                cor = False
                try:
                    cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
                    if isinstance(cor_fname, list) and len(cor_fname) == 1 :
                        com = pd.read_csv(cor_fname[0]).set_index('time')
                        print(f'\nLoaded an Al-corrected CoM trajectory for {variable}={var}, {variable2}={var2} T1={T1}K')
                        cor = True
                except : continue
                
                dtt = com.index[1]-com.index[0]
                
                ## average multiple starts
                for c, duration in enumerate(durations): 
                    
                    ## enforce start_step >= duration 
                    if enforce_random and start_step < duration : start_step = duration
                    
                    if com.index.max() <= duration*4 :
                        dcoms.append(np.nan)
                        # print(f'Clipping long duration: {variable}={var}, {variable2}={var2} T1={T1}K')
                        continue
                
                    dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
                    # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
                    dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
                    
                    all_dcom2 = dr.dcom.values / 4
                    
                    dcoms.append(np.mean(all_dcom2))
                    
                    ## divide by st.dev. / plot D_CoM rescaled by its st.dev.
                    # print(f'{mm} {cn} {T1}K : mean {np.mean(all_dcom2):.3g} A2/ps, st.d. {np.std(all_dcom2):.3g} A2/ps (lag={duration}, step={start_step}), simple method')
                    if rescale : all_dcom2 = np.array(all_dcom2) / np.std(all_dcom2)
                
                    ## plot stuff                
                    if option == 'hist' :
                        ## plot a histogram of D_com
                        dcom2_bins = np.linspace(min(all_dcom2)*(0.99 if min(all_dcom2) < 0 else 1.01), max(all_dcom2)*1.01,50)
                        hist, bin_edges = np.histogram(all_dcom2, bins=dcom2_bins, density=True)
                        bin_ctrs = (bin_edges[:-1] + bin_edges[1:])*0.5
                        
                        if duration < 10 :
                            l = f'{duration:.1f} ps'  
                        elif duration > 1e3 :
                            l = f'{duration/1e3:.1f} ns' 
                        else : 
                            l = f'{duration:.0f} ps'
                            
                        ax.plot(bin_ctrs, hist, label=f'$t=${l}', linestyle='-', marker=None)
                        
                del com
                
                ax.legend(title=f'{mm} {phases[ph]}, {T1}K')

## axes decorations
for ax in axes: 
    if rescale :
        ax.plot([1,2,3],[0.1,0.1/np.e, 0.1/np.e**2], c='k', lw=0.4)
        ax.set(xlim=[0,9.5], xlabel=r'$D_{CoM}(t;\Delta,\delta)/\sigma_{D_{CoM}}$')
        axes[0].set(ylabel=r'$P(D_{CoM})$, $\sigma^{-1}_{D_{CoM}}$', yscale='log')
    else : 
        ax.set(xscale='log', yscale='log', ylim=[0.5,None])
        axes[0].set(ylabel=f'$P(D_{{CoM}}(t;\Delta,\delta={start_step}ps)$, ps $\AA^{{-2}}$')
        ax.set(xlabel=r'$D_{CoM}(t;\Delta,\delta)$, $\AA^2$/ps')
        
    ## recolor lines
    non_guide_lines = [x for x in ax.lines if x.get_label()[0] != '_']
    colors = [batlow_even(j) for j in np.linspace(0, 1, len(non_guide_lines))]
    for i, l in enumerate(non_guide_lines) : l.set(color=colors[i])
    
    ## remake legend with same title - but possibly new colors
    ax.legend(title=ax.get_legend().get_title().get_text(), loc='upper right' if rescale else 'lower left')

fig.tight_layout(pad=0.5, w_pad=0.25)

# =============================================================================
# %% Figure SN : Crowding by Oi 
# =============================================================================

# ===== load data =====

planes_to_load = all_planes.query('metal == "Na" & num_planes > 1 & phase == "beta" & T1 in [300,600] & config in ["120_1", "120_4"]')
planes_to_load = all_planes.query('metal == "Na" & num_planes > 1 & phase == "beta" & T1 == 300 & config in ["120_M1", "120_M4"]')

# ===== parameters =====

option = 'spectra'  ## 'spectra' for conductivity, 'hist' for distributions of DCOM, 
start_1 = 0         ## [ps] time at which to start sampling CoM MSD
start_step = 10    ## [ps] interval for sampling CoM MSD  

durations = np.round(np.logspace(0.4,3.4),2)     ## [ps] how long each sampling is. Time "Delta" from Barkai

enforce_random = False ## flag to enforce start_step >= duration 

# ========== automatic things below this line ==========

variable = 'T1' ## hard-coded here for paper figures
variable2 = 'config'

macro_planes_data = hu.load_macro_planes(planes_to_load).sort_values(by=[variable2,variable])

## values of the first variable in the loaded data
var_values = sorted(macro_planes_data[variable].unique())
var2_values = sorted(macro_planes_data[variable2].unique())
colors = cycle([batlow_even(i) for i in np.linspace(0, 1, len(var2_values))])


# ===== conductivity =====

## figure to plot the distributions
## do not share x-scale if each axes is at a different temperature ## sharex=(variable != 'T1'),
fig, ax = plt.subplots(figsize=(5,3.75))

## loop over the values of the variables
for var in var_values :
    for var2 in var2_values:
        
        plane = macro_planes_data.query(f'{variable} == @var & {variable2} == @var2').iloc[0]
                
        ph = plane.phase; mm = plane.metal; cn = plane.config
        st = plane.stoich; ex = plane.exclude; T1 = plane.T1
        N = len(plane.atoms.query('atom == @mm'))
        cell = plane.cell
        
        defect_type = [r'$O_i^{\prime\prime}$', r'$Mg_{Al}^\prime$'][('M' in ex) or (ph == 'bdp')]
        
        dcoms = list()
        
        ## load a pre-corrected result
        cor = False
        try:
            cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
            if isinstance(cor_fname, list) and len(cor_fname) == 1 :
                com = pd.read_csv(cor_fname[0]).set_index('time')
                print(f'\nLoaded an Al-corrected CoM trajectory for {variable}={var}, {variable2}={var2} T1={T1}K')
                cor = True
        except : com = None
            
        ## check that the trajectory is loaded, and is long enough
        # if (com is None) or (com.index.max() <= duration) : 
        if com is None :
            print(f'CoM trajectory too short. Found {com.index.max()} ps, need {duration} ps.\n')
            continue
        
        dtt = com.index[1]-com.index[0]
        
        ## average multiple starts
        for duration in durations: 
            
            ## enforce start_step >= duration 
            if enforce_random and start_step < duration : start_step = duration
            
            if com.index.max() <= duration*4 :
                dcoms.append(np.nan)
                print(f'Clipping long duration: {variable}={var}, {variable2}={var2}')
                continue
        
            dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
            # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
            dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
            
            all_dcom2 = dr.dcom.values * N / 4
            
            dcoms.append(np.mean(all_dcom2))
                
        sigts = sigma_T(N,cell,np.array(dcoms)*unit_conv)
        l = f'{mm}$_{{{eval(st)/100:.2f}}}~\sigma$ from $r_{{CM}}^2 $' if len(macro_planes_data) == 1 and ph == 'beta' else f'{var2}{units[variable2]}'
        l = f'{mm}$_{{{st}}}~\sigma$ from $r_{{CM}}^2 $' if len(macro_planes_data) == 1 and ph != 'beta' else l
        l = f'{T1}K, {defect_type} {ex.replace("M","")}+'
        ax.plot(1e12/durations, sigts/T1, next(markers)+next(lines), 
                label=l, mfc='none', c=next(colors))
        
        del com
                
ax.legend(loc='lower right', title=f'{mm} {phases[ph]}-alumina')
ax.set(yscale='log', xscale='log', xlabel=r'$\nu=1/t$, Hz') 
ax.set(ylabel=r'$\sigma_{xy}(t;\Delta)$, S/cm')
ax.set(xlim=[1e8,6e11])

fig.tight_layout(pad=0.5, h_pad=0.25)
        
# ===== r2 and C_D =====

dim = 2

## new figure with 2 panels
fig3, axes = plt.subplots(3,1, sharex=True, figsize=(4.5, 9.9))

## loop over the values of the variables
for var in var_values :
    for var2 in var2_values:
        
        plane = macro_planes_data.query(f'{variable} == @var & {variable2} == @var2').iloc[0]
                
        ph = plane.phase; mm = plane.metal; cn = plane.config
        st = plane.stoich; ex = plane.exclude; T1 = plane.T1
        N = len(plane.atoms.query('atom == @mm'))
        cell = plane.cell
        
        defect_type = [r'$O_i^{\prime\prime}$', r'$Mg_{Al}^\prime$'][('M' in ex) or (ph == 'bdp')]
        label = f'{T1}K, {defect_type} {ex.replace("M","")}+'
        leg_title = f'{mm} {phases[ph]}-alumina'
        this_color = next(colors); this_line = next(lines)
        
        ## load the 2D a2 file - leaving out the "split" files
        a2_xys = glob.glob(plane.folder+f'/{mm}*{ex}-a2xy*{T1}K*ps.csv')

        ## load the a2 file if exactly one exists, else complain
        if a2_xys :
            if len(a2_xys) > 1 : a2_xys = sorted(a2_xys, reverse=True,
                            key = lambda x : eval(re.split('-|_| ',x)[-1][:-6]))

            ## read the a2 file - options for csv file
            a2 = pd.read_csv(a2_xys[0], sep=',').set_index('time')
        else : print(f'could not load a 2D a2 file for plane {mm} {cn} {T1}K')
        
        ## recalculate a2 for the right number of dimensions
        a2.a2 = dim * a2.r4 / a2.r2 ** 2 / (dim+2) - 1
    
        ## load a short-time a2 file if using one
        try : 
            a2s = pd.read_csv(plane.folder + f'/{mm}-{st}-{ex}-a2{"xy" if dim == 2 else ""}-{T1}K-10ps.csv').set_index('time')
            a2s.a2 = dim * a2s.r4 / a2s.r2 ** 2 / (dim+2) - 1
        except : 
            print(f'could not load a short a2 file for plane {mm} {cn} {T1}K')
            a2s = None
        
        ## load Gs 
        ## Plot points from van Hove function
        try : gs = hu.load_gs(plane.folder+f'/{mm}-*-gs-{T1}K*ps.csv', 'cdt', radii=[[0.01,1.7],[0.01,4.6]])
        except ValueError : print(f'something wrong with Gs for {mm} {cn} {T1}, check fractional/real computation.')
        
        # === r2 ===
        
        axes[0].plot(a2.r2.iloc[1:], label=label, c=this_color, ls=this_line)
        axes[0].plot(a2s.r2.iloc[1:].loc[:0.02], c=this_color, ls=this_line)
        axes[0].legend(title=leg_title, loc='upper left')
        
        # === NGP ===
        
        axes[1].plot(a2.a2, label=label, c=this_color, ls=this_line)
        axes[1].plot(a2s.a2.iloc[1:].loc[:0.02], c=this_color, ls=this_line)
                    
        ## create the interpolator for plotting little stars based on Gs
        int_fun = interp1d(a2.index.values, a2.a2)
    
        ## Plot points from van Hove function
        axes[1].plot(gs, int_fun(gs), marker='o', mfc='yellow', ls='', markersize=4,
                        mec='k', zorder=3, fillstyle='full')
                
        axes[1].legend(title=leg_title, loc='lower left')
        
        # === C_D ===
        
        start = dt.now()
        svals = np.logspace(-5, 2, 4000) # if not short else np.logspace(-6,5,3000)
        
        ## Laplace transform of C_D(t)
        cds = hu.fluctuation_kernel(a2, svals, dim=dim)
        
        try: cdt = hu.stehfest_inverse(cds, a2.index.values[1:-1])
        except :
            print(f'could not append inverse transform for {mm} {cn} {T1}')
            break
        cdt = pd.DataFrame({'time':a2.index.values[1:-1],'cdt':cdt}).set_index('time')
        axes[2].plot(cdt.cdt.loc[0.2:a2.index.max()/3+1], label=label, c=this_color, ls=this_line) 
        
        ## create the interpolator for plotting relaxation times based on Gs
        int_fun = interp1d(cdt.index.values, cdt.cdt)
        try: axes[2].scatter(gs, int_fun(gs), marker='o', facecolors='yellow', edgecolors='k', zorder=3, s=16)
        except : print('something wrong with plotting Gs * for {mm} {cn} {T1}')
        
        ## plot short-time separately
        try: 
            cds_s = hu.fluctuation_kernel(a2s, np.logspace(0,4,1000), dim=dim)
            cdt_s = hu.stehfest_inverse(cds_s, a2s.index.values[1:-1])
            cdt_s = pd.DataFrame({'time':a2s.index.values[1:-1],'cdt':cdt_s}).set_index('time')
            axes[2].plot(cdt_s.cdt.loc[0.005:0.2], ls=this_line, c=this_color) 
        except : print(f'could compute short-time C_D for plane {mm} {cn} {T1}K')
        
        print(f'done {T1}K, time taken {(dt.now()-start).total_seconds():.2f}')
        
        axes[2].legend(title=leg_title, loc='upper left')
        
axes[0].set(xscale='log', yscale='log', xlim=[0.005, 5e3], ylim=[5e-3, 2e3],
            ylabel=r'$\langle \overline{r^2(t)} \rangle$, $\AA^2$',
            yticks=[0.1,1,10,100], yticklabels=['0.1','1.0', '10', r'$10^2$'])
axes[1].set(ylabel='Non-Gaussian Parameter', ylim=[0.08, 15], yscale='log',
            yticks=[0.1,1,10], yticklabels=['0.1','1.0', '10'])
axes[2].plot([1e-3,1e5],[0,0], lw=0.4, c='grey', ls=':')
axes[2].set(xlabel='Time lag $t$, ps', ylabel=r'$C_D(t)$')

for i in range(5) : fig3.tight_layout(pad=0.5, h_pad=0.25)
        
# =============================================================================
# %% Figure 6: Quenching and crowding
# =============================================================================

# ===== parameters =====

option = 'spectra'  ## 'spectra' for conductivity, 'hist' for distributions of DCOM, 
start_1 = 0         ## [ps] time at which to start sampling CoM MSD
start_step = 10    ## [ps] interval for sampling CoM MSD  

durations = np.round(np.logspace(0.4,3.4),2)     ## [ps] how long each sampling is. Time "Delta" from Barkai

enforce_random = False ## flag to enforce start_step >= duration 

# ========== automatic things below this line ==========

colors = cycle([batlow_even(i) for i in np.linspace(0, 1, 2)])

fig, axes = plt.subplots(2,1,figsize=(4,7.5), sharex=True)

# ===== load data: bdp =====

planes_to_load = all_planes.query('metal == "Na" & num_planes > 1 & phase == "bdp" & T1 == 300 & config in ["symm_1", "unsym_0"]')
macro_planes_data = hu.load_macro_planes(planes_to_load).sort_values(by=['config',"T1"])

# ===== conductivity: bdp =====

## loop over planes
for i, plane in macro_planes_data.iterrows():
    
    # plane = macro_planes_data.query(f'{variable} == @var & {variable2} == @var2').iloc[0]
            
    ph = plane.phase; mm = plane.metal; cn = plane.config
    st = plane.stoich; ex = plane.exclude; T1 = plane.T1
    N = len(plane.atoms.query('atom == @mm'))
    cell = plane.cell
    
    defect_type = ['not quenched (Fig. 3)', 'quenched']['symm' in cn]
    
    dcoms = list()
    
    ## load a pre-corrected result
    cor = False
    try:
        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
            com = pd.read_csv(cor_fname[0]).set_index('time')
            print(f'\nLoaded an Al-corrected CoM trajectory for {variable}={var}, {variable2}={var2} T1={T1}K')
            cor = True
    except : com = None
        
    ## check that the trajectory is loaded, and is long enough
    # if (com is None) or (com.index.max() <= duration) : 
    if com is None :
        print(f'CoM trajectory too short. Found {com.index.max()} ps, need {duration} ps.\n')
        continue
    
    dtt = com.index[1]-com.index[0]
    
    ## average multiple starts
    for duration in durations: 
        
        ## enforce start_step >= duration 
        if enforce_random and start_step < duration : start_step = duration
        
        if com.index.max() <= duration*4 :
            dcoms.append(np.nan)
            print(f'Clipping long duration: {variable}={var}, {variable2}={var2}')
            continue
    
        dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
        # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
        dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
        
        all_dcom2 = dr.dcom.values * N / 4
        
        dcoms.append(np.mean(all_dcom2))
            
    sigts = sigma_T(N,cell,np.array(dcoms)*unit_conv)
    l = f'{mm} {phases[ph]} {T1}K, {defect_type} '
    axes[0].plot(1e12/durations, sigts/T1, next(markers)+next(lines), 
            label=l, mfc='none', c=next(colors))
    
    del com

# ===== load data: beta =====

planes_to_load = all_planes.query('metal == "Na" & num_planes > 1 & phase == "beta" & T1 in [300,600] & config in ["120_1", "120_4"]')
macro_planes_data = hu.load_macro_planes(planes_to_load).sort_values(by=['T1', 'config'])

## loop over planes
for i, plane in macro_planes_data.iterrows():
    
    # plane = macro_planes_data.query(f'{variable} == @var & {variable2} == @var2').iloc[0]
            
    ph = plane.phase; mm = plane.metal; cn = plane.config
    st = plane.stoich; ex = plane.exclude; T1 = plane.T1
    N = len(plane.atoms.query('atom == @mm'))
    cell = plane.cell
    
    defect_type = [r'$O_i^{\prime\prime}$', r'$Mg_{Al}^\prime$'][('M' in ex) or (ph == 'bdp')]
    
    dcoms = list()
    
    ## load a pre-corrected result
    cor = False
    try:
        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
            com = pd.read_csv(cor_fname[0]).set_index('time')
            print(f'\nLoaded an Al-corrected CoM trajectory for {variable}={var}, {variable2}={var2} T1={T1}K')
            cor = True
    except : com = None
        
    ## check that the trajectory is loaded, and is long enough
    # if (com is None) or (com.index.max() <= duration) : 
    if com is None :
        print(f'CoM trajectory too short. Found {com.index.max()} ps, need {duration} ps.\n')
        continue
    
    dtt = com.index[1]-com.index[0]
    
    ## average multiple starts
    for duration in durations: 
        
        ## enforce start_step >= duration 
        if enforce_random and start_step < duration : start_step = duration
        
        if com.index.max() <= duration*4 :
            dcoms.append(np.nan)
            print(f'Clipping long duration: {variable}={var}, {variable2}={var2}')
            continue
    
        dr = com.loc[duration+com.index.min()::int(start_step/dtt)] - com.loc[:com.index.max()-duration:int(start_step/dtt)].values
        # dr['dcom'] = (dr.x**2 + dr.y**2 + dr.z**2)/duration
        dr['dcom'] = (dr.x**2 + dr.y**2 )/duration
        
        all_dcom2 = dr.dcom.values * N / 4
        
        dcoms.append(np.mean(all_dcom2))
            
    sigts = sigma_T(N,cell,np.array(dcoms)*unit_conv)

    l = f'{T1}K, {defect_type} {ex.replace("M","")}+'
    axes[1].plot(1e12/durations, sigts/T1, next(markers)+next(lines), 
            label=l, mfc='none', c=next(colors))
    
    del com
    
axes[1].set(xscale='log', xlim=[1e8,8e11], xlabel=r'$\nu=1/t$, Hz')
    
for ax in axes :
    ax.set(ylabel=r'$\sigma_{xy}(t;\Delta)$ from eMSD, S/cm', yscale='log', ylim=[2e-3,4])
    
# =============================================================================
# %% Center of mass (CoM) displacement with multiple start sampling
# ##          and ionic conductivity from that
# =============================================================================

## variable by which to plot stuff
variable = 'metal' ## << pick a variable
start_1 = 0         ## [ps] time at which to start sampling CoM MSD
start_step = 2500    ## [ps] interval for sampling CoM MSD
start_last = 97500    ## [ps] last time at which to sample CoM MSD
  
duration = 2500     ## [ps] how long each sampling is

# ========== automatic things below this line ==========

## array for multiple starts and its tuple description for lookup of pre-dones
starts = np.arange(start_1,start_last,start_step,dtype=float) 
spec=(start_1,start_step,start_last,duration)  

## values of the first variable in the loaded data
var_values = sorted(macro_planes_data[variable].unique())

## initialize a second variable
variable2 = None

## deduce a second variable
if len(macro_planes_data) > len(var_values) * len(macro_planes_data.T1.unique()): ## a second variable is varied
    for rv in [x for x in relevant_vars if x != 'T1' and x != 'z']: 
        if rv in macro_planes_data.columns and len(set(macro_planes_data[rv])) > 1 and rv != variable: 
            variable2 = rv
            break
else: variable2 = 'config'
var2_values = sorted(macro_planes_data[variable2].unique())
        
## try pre-loading and filtering for the same computation conditions right away
try: 
    sigmas_msd = pd.read_csv('./production/sigmas_msd.csv')
    sigmas_msd.spec = sigmas_msd.spec.apply(lambda x : eval(x))        
    sigmas_msd_spec = sigmas_msd.query('spec == @spec')
except : sigmas_msd = pd.DataFrame()

## structure for new computations
new_sigmas_msd = list()

## loop over the values of the variables
for var in var_values :
    for var2 in var2_values:
        for i, plane in macro_planes_data.iterrows():
            if plane[variable] == var and plane[variable2] == var2:
                
                all_dcom = list()
                all_derr = list()
                ph = plane.phase
                mm = plane.metal
                cn = plane.config
                st = plane.stoich
                ex = plane.exclude
                T1 = plane.T1
                com = plane.com
                N = len(plane.atoms.query('atom == @mm'))
                
                ## load a pre-computed result
                try:
                    ## check this configuration was already computed
                    pre_done = sigmas_msd_spec.query('metal == @mm & stoich == @st & exclude == @ex & T1 == @T1')
                    
                    assert len(pre_done) > 0 
                    new_sigmas_msd.append(pre_done.to_dict('records')[0])
                    print(f'pre-done {variable}={var}, {variable2}={var2} T1={T1}K')
                    print(f'sigma*T = {pre_done.sigt.iloc[0]:.2e} S*K/cm')
                    print(f'±1 standard deviation: {pre_done.sigt_1sm.iloc[0]:.2e} - {pre_done.sigt_1sp.iloc[0]:.2e} \n')
                    continue
                
                ## nothing pre-computed, needs to be computed
                except: 
                    
                    ## load a pre-corrected CoM trajectory
                    cor = False
                    try:
                        cor_fname = glob.glob(plane.folder + f'/cm_{mm}-{st}-{ex}-{T1}K-{mm}-cor.csv')
                        if isinstance(cor_fname, list) and len(cor_fname) == 1 :
                            com = pd.read_csv(cor_fname[0]).set_index('time')
                            print(f'\nLoaded an Al-corrected CoM trajectory for {variable}={var}, {variable2}={var2} T1={T1}K')
                            cor = True
                    except : com = None
                
                ## average multiple starts
                print(f'computing {variable}={var}, {variable2}={var2} T1={T1}K\n')
                for i, s in enumerate(starts):
                    com_s = com.loc[ (s < com.index) & (com.index < s + duration)]
                    if len(com_s) < 1 : continue
                    com_s -= com_s.iloc[0]
                    com_s['r2'] = com_s.x**2 + com_s.y**2 # + com_s.z**2
                    
                    try: 
                        al_s = al.loc[ (s < al.index) & (al.index < s + duration)]
                        al_s -= al_s.iloc[0]
                        # al_s['r2'] = al_s.x**2 + al_s.y**2 # + com_s.z**2
                        # print(com_s.tail(1).values/al_s.tail(1).values)
                        com_s['r2'] = (com_s.x-al_s.x)**2 + (com_s.y-al_s.y)**2 # + (com_s.z-al_s.z)**2
                    except: pass
                    
                    ## linear fit
                    com_s = com_s.dropna()
                    popt, pcov = cf(linfit, com_s.index.values, com_s.r2.values, p0=[0,0])
                    perr = np.sqrt(np.diag(pcov))
                    
                    ## convert [AA^2/ps] units to normal units [cm^2/sec]
                    d_com, d_err = [unit_conv*x[0] for x in [popt, perr]]
                    
                    ## add diffusion coeff to average
                    ## NB: mulitplication by N and division by 2*dimensions
                    all_dcom.append(d_com*N/4)
                    all_derr.append(d_err*N/4)
                    
                    ## output in case it takes a while; ## and len(starts) > 50
                    if not i % 20 : print(f'computed starting point {s} ps')
                    
                    del com_s
                
                ## this might need an extra call to np.mean()
                sigt_10 = sigma_T(N,cell,np.percentile(all_dcom,10))
                sigt_1sm = sigma_T(N,cell,np.percentile(all_dcom,50-34.1))
                sigt_20 = sigma_T(N,cell,np.percentile(all_dcom,20))
                sigt = sigma_T(N,cell,np.mean(all_dcom)) 
                sigt_80 = sigma_T(N,cell,np.percentile(all_dcom,80))
                sigt_1sp = sigma_T(N,cell,np.percentile(all_dcom,50+34.1))
                sigt_90 = sigma_T(N,cell,np.percentile(all_dcom,90))
                sigt_err = sigma_T(N,cell,np.std(all_dcom))
                
                print(f'\n===== {variable}={var}, {variable2}={var2} T1={T1}K =====')
                print(f'D_com = {np.mean(all_dcom):.2E} ± {np.std(all_dcom):.2E} cm2/sec')
                print(f'Avg error in fits = {np.mean(all_derr):.2E}')
                print(f'sigma*T = {sigt:.2e} ± {sigt_err:.2E} S*K/cm')
                print(f'20th-80th percentiles: {sigt_20:.2e} - {sigt_80:.2e}')
                print(f'±1 standard deviation: {sigt_1sm:.2e} - {sigt_1sp:.2e}')
                print(f'10th-90th percentiles: {sigt_10:.2e} - {sigt_90:.2e} \n')
                
                new_sigmas_msd.append(dict(phase=ph, metal=mm, config=cn, stoich=st, exclude=ex, z='z_all',
                                   T1=plane.T1, spec=spec, sigt=sigt, sigt_1sm=sigt_1sm, sigt_1sp=sigt_1sp,
                                   sigt_10=sigt_10, sigt_20=sigt_20, sigt_80=sigt_80, sigt_90=sigt_90))
                del com
                
## convert all fitted sigmas to dataframe
new_sigmas_msd = pd.DataFrame(new_sigmas_msd)

## merge new things into the overall dataframe and re-save
sigmas_msd = sigmas_msd.append(new_sigmas_msd, ignore_index=True).drop_duplicates(subset=['metal','config','stoich','exclude','T1','z','spec'])
sigmas_msd.to_csv('./production/sigmas_msd.csv', index=False, float_format='%.7g')
try : del sigmas_msd, sigmas_msd_spec
except: pass  








