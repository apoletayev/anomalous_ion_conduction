#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andreypoletaev
"""
# =============================================================================
# %% Block 1: initial imports & constants
# =============================================================================

import os, sys, re, freud

if os.path.join(os.path.abspath(os.getcwd()), "utils") not in sys.path :
    sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "utils"))

import numpy as np
import pandas as pd
import seaborn as sns
import hop_utils as hu
import crystal_utils as cu

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from datetime import datetime as dt
from itertools import cycle

from batlow import cm_data
batlow_cm = LinearSegmentedColormap.from_list('batlow', cm_data)
batlow_even = LinearSegmentedColormap.from_list('batlow_even', hu.batlow_lightness_scaled(0.4,0.6))

plt.rc('legend', fontsize=10)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16)
plt.rc("savefig", format='pdf')
plt.rc('markers', fillstyle='full')
plt.rc('errorbar', capsize=3)

kb = 8.617e-5 ## [eV/K]

dddddd = dt.today().strftime('%y%m%d')

# =============================================================================
#  Constants
# =============================================================================

## relevant_vars are actually used as of 2020/07/26
relevant_vars = ['metal','phase','T1','z','config','stoich','exclude']

## default number of sites per plane (beta)
num_polys = 200 

## estimated maximum distances from oxygens based on stoichiometry
est_max_radii = {'100':0, '102':15, '106':9, '116':5, '120':5, '124':4, '130':4, 'unsym':6, 'symm':6, 'oh':6}

## all the possible mobile ions
mobile_ions = ['Na', 'K', 'Ag']

## accepted names for doubleprime
bdp_names = ['beta', 'doubleprime', 'bdp']

## doubleprime stoichiometries
bdp_stoichs = ['symm', 'unsym', 'oh', 'rand']

## figure display name of the distance metric by phase 
distance_names = dict()
for n in bdp_names:
    distance_names[n] = r'# Mg$_{Al}^\prime$ Neighbors'
distance_names['beta']=r'Distance to $O_i^{\prime\prime}$'

# empty placeholder
planes_data = None

## shorthands for labels in figures
bdp =  r'$\beta^{\prime\prime}$'
beta = r'$\beta$'
phases = {'beta':beta, 'bdp':bdp}

## database of all the hops
try: all_planes = pd.read_csv('./sample_data/all_hop_planes.csv')
except: pass

## flag for loading atoms
frac = False

## flag for loading site-filling times (used in the PDF plotting block)
do_fill_times= False 

## clear the data structure that holds the heavy hops dataframes
try: del planes_data
except : pass

# =============================================================================
# %% Block 1: Compile a database of files each of which stores hop events.
# ## This only screens for files that have 'plane.csv' in their name
# ## (Currently planes are organized into folders where they share a folder
# ## with only one lammps structure file. Crude, I know.)
# ## 
# ## Currently this is pretty loose, but can be made stricter, e.g. looking for 
# ## some identifiers to appear both in the filename AND in the folder name.
# =============================================================================

red_flags = ['grid', 'VACF', 'polymer', 'LAPC', 'legacy', 'verts', 
             'hop_planes', 'vanHove', 'BR', 'fill_times', 'actual',
             'count', 'expected','Murch', 'paths', 'NN', 'counts', 'LAMMPS',
             'oxygen_cells', 'edge', 'pre-update', 'jacf', 'vacf']

# list of dictionaries from which a pandas dataframe will be composed
dicts = []

for root, dirs, files in os.walk('./sample_data/'):
    for f in files:
        fp = os.path.join(root,f)
        if f.endswith('plane.csv') and hu.which_one_in(red_flags, fp) is None :

            # print(re.split('/|-', root), f)
            
            ## split stuff for finding all the specific variables
            features = list(set(re.split('-|/',fp)))
            
            ## phase: beta or beta"
            ph = hu.which_one_in(bdp_names, features)
            if ph is None or not ph: 
                print(f'Phase unclear: {fp}')
                continue
            
            ## temperature: look for decimals followed by 'K'
            temps = set(re.findall(r'\d+K', fp))
            if len(temps) != 1 : 
                print(f'Temperature unclear: {fp}')
                continue
            else: 
                # print(f'found temperature: {list(temps)[0]}')
                this_temp = int(list(temps)[0][:-1])
            
            ## metal : look for there to be one mobile ion only, somewhere in the path
            mobile_ion = hu.which_one_in(mobile_ions, features)
            if mobile_ion is None or not mobile_ion:
                print(f'Mobile ion unclear: {fp}')
                continue
            
            ## stoichiometry: works for the beta phase 2020/07/26; bdp added 08/01
            if ph == 'beta' :
                stoichs = set(cu.flatten([re.findall(r'1\d\d_\d|1\d\d$|1\d\d_M\d.|1\d\d_M\d', feat) for feat in features]))
                if len(stoichs) != 1:
                    print(f'Composition unclear: {fp}')
                    print(f'Culprit: {stoichs}')
                    continue
                else:
                    lammps_stoich = list(stoichs)[0]
                    if '_' in lammps_stoich: st, ex = lammps_stoich.split('_')
                    else:
                        st = lammps_stoich
                        ex = 0
            ## for not-beta phase, assume it is beta-doubleprime
            else:
                stoich = [x for x in features if hu.which_one_in(bdp_stoichs, x)]
                if len(stoich) > 1 :
                    print(f'Beta-doubleprime "rule" unclear: {stoich}.')
                    print('Recognized rules: "oh" for Al(4), "unsym" for un-symmetric Al(2),')
                    print('                  "symm" (2Ms!) for any Al(2), "rand" for all sites.')
                    print('To fix this: rename the folders and/or the lammps file so that they all match.')
                    continue
                else:
                    lammps_stoich = stoich[0]
                    if '_' in stoich[0]: st, ex = stoich[0].split('_')
                    else: st = stoich[0]; ex = 0
                    
            ## find the lammps file: there should only be one 
            ## this is not really robust as it does not test for whether the file is real
            lf = '/'.join([x for x in root.split('/') if 'hops' not in x]) + '/'
            lp = lf + mobile_ion + ph + lammps_stoich + '.lmp'
            try:
                with open(lp) : pass
            except IOError : 
                try:
                    lp = lf + mobile_ion + '_' + ph + '_' + lammps_stoich + '.lmp'
                except IOError :
                    print(f'Could not find lammps file: {lp}')
                    continue
            
            ## z of the plane. 
            ## This is not great: it relies on having a temperature before z in the file name
            ## The value 4 is also not robust to beta vs doubleprime. 
            features2 = list(re.split('K_|-|/|\.',fp))
            zs = [x.replace('plane','') for x in features2 if 'plane' in x]
            if len(zs) != 1:
                print(f'z-value unclear: {zs} in {fp}')
                continue
            else:
                zs = zs[0]
                if zs == 'z_all' :
                    _, _, _, atoms = cu.read_lmp(lp)
                    num_p = len(cu.get_conduction_planes(atoms,mobile_ion,
                                    inexact=False if ph == 'beta' else True))
                else: num_p = 1
                
            ## time length of simulation
            hops = pd.read_csv(fp)
            one_ion_hops = hops.query(f'ion == {hops.ion.min()}')
            if len(one_ion_hops) > 1:
                total_time = one_ion_hops.new_resid_time.sum() + one_ion_hops.old_resid_time.iloc[0]
            else: 
                total_time = one_ion_hops.new_resid_time.values[0]
            print(f'logged plane {fp}')
            del hops, one_ion_hops
            
            ## put it all together
            dicts.append(dict(phase=ph, stoich=st, exclude=ex, config=lammps_stoich, metal=mobile_ion,
                              T1=this_temp, z=zs, num_planes=num_p, total_time=total_time,
                              hop_path=fp, lammps_path=lp))
    
## sort and save
planes_df = pd.DataFrame(dicts).sort_values(by=['phase','metal','config','T1','z'], axis=0)   
planes_df.to_csv('./sample_data/all_hop_planes.csv', index=False)

# =============================================================================
# %% Block 2: combine planes based on lammps files: iterate through 
# ## all lammps files and combine planes if:
# ## (a) there is not a 'z_all' plane and 
# ## (b) all needed single planes exist
# ## NOTE: this does not deal with there being multiples of planes at a same z
# ## in the same folder, they just get added together. 
# ## This is not good, because to do it properly will require re-naming ions.
# ## IMPORTANT:
# ## After running this (if generating new z_all planes), re-run the previous 
# ## block to actually add them to the "registry"
# =============================================================================

all_planes = pd.read_csv('./sample_data/all_hop_planes.csv')

lammps_files = all_planes.lammps_path.unique()

for lf in lammps_files:
    
    ## query one lammps file
    this_file = lf.split('/')[-1]
    one_file_planes = all_planes.query('lammps_path == @lf')
    mm = one_file_planes.metal.unique()[0]
    
    ## load the lammps file (default is fractional coordinates)
    try: 
        print(f'\nLoading {this_file}')
        _, _, cell, atoms = cu.read_lmp(lf)
        
        ## ignore oxygen defects: remove the max() of types that are oxygen
        if len( atoms.query('atom == "O"').type.unique() ) > 1:
            type_ointer = atoms.query('atom == "O"').type.max()
            atoms = atoms.query('type != @type_ointer')
            
        ## get all conduction planes and the number of sites in them
        planes = cu.get_conduction_planes(atoms,mm,inexact=False if 'beta' in lf else True)
        site_pts = cu.get_mobile_ion_sites(atoms, planes[0])
        polys = len(site_pts)
        print(f'File has {polys} sites per plane')
    except:
        print(f'could not load lammps file: {lf}')
        continue
    
    ## get all possible z coordinates for planes in this lammps file
    planes = [cu.standard_plane_name(p) for p in planes]
    
    ## iterate over temperatures
    for T1 in one_file_planes.T1.unique():
        
        ## get all planes by z at this temperature
        T1_planes = one_file_planes.query('T1 == @TK')
        # print(type(T1_planes.z.unique()[0]), T1_planes.z.unique())
        these_zs = [f'{int(x):03d}' if hu.s2n(str(x)) else x for x in T1_planes.z.unique()]
    
        ## check for a complete combined plane
        if 'z_all' in these_zs :
            print(f"Already combined planes for {this_file} at T1 = {T1:4d}K.")
            continue
        
        ## if the planes are not already combined, check that all planes have hop files
        else:
            if sum([p in these_zs for p in planes]) == len(planes):
                ## looks like hops in all planes are available. combine planes 
                print(f"Combining planes for {this_file} at T1 = {T1:4d}K.")
                planes_list = T1_planes.hop_path.values.tolist()
                planes_folder = '/'.join(planes_list[0].split('/')[:-1])
                zs_list = T1_planes.z.values.tolist()
                combined = hu.combine_planes3(planes_list, zs_list, numpolys=polys, verbose=True)
                combined_path = planes_folder + '/' + mm + str(T1) + 'K_z_allplane.csv'
                combined.to_csv(combined_path, index=False)


# =============================================================================
# %% Figure 5: pre-load data so it does not take time every time
# =============================================================================
    
planes_to_plot = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 in [300,600]')
# planes_to_plot = all_planes.query('phase == "beta" & num_planes > 3 & config == "120_4" & T1 == 600')

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac=False, do_fill_times=False)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)

# =============================================================================
# %% Figure 5: Plot semi-log of onward/reverse hopping ratios as correlation factor
# ## This is the direct comparison with Funke's model
# =============================================================================

variable = 'metal' ## << do pick this

## distance of the furthest sites away from defects that will be plotted
## (larger values will be ignored, if they are present)
max_r_to_plot = 4

clb_cut = 10    ## CLB: min Count (of hops) per Log Bin to plot; smaller counts (at longer times) get cut

old_site = 'all' ## plot for hops starting at sites: 'all', 'aBR', or 'BR'

full_label = False    ## cosmetic changes to labels of lines
plot_total = True    ## plot for all distances together

shade_range = True   ## shade between top and bottom curves for each material

logbins = np.logspace(-0.7,4.3)  ## sigma np.round(np.logspace(-0.7,4.3),2)

plot_gs = True          ## plot relaxation times from van Hove function (self)
rs_list = [[0.01, 4.6]] ## default radii for van Hove decay points

plot_cdt = True ## add C_D values to the plot

# ========== automatic things below this line ==========

plt.rc('markers', fillstyle='full')

# ===== start figure =====

logcenters = np.sqrt(logbins[1:]*logbins[:-1])

var_values = sorted(set(planes_data[variable]))

fig, axes = plt.subplots(1,len(var_values),
                         sharex=True, sharey=True,figsize=(6*len(var_values),4.8))

if len(var_values) < 2 : axes = [axes]

# ===== iterate over the loaded planes =====

for var, ax in zip(var_values, axes):
    
    # these_planes = planes_data.loc[planes_data[variable]==var]
    these_planes = planes_data.query(f'{variable} == @var')
    
    ## start counters for colors and markers afresh for each panel
    color_counter = 0
    markers = cycle(['o', 's', 'v', 'd', '^','D'])
    
    for i, plane in planes_data.iterrows():
        if plane[variable] == var:
            
            m = plane.metal; T1 = plane.T1; z = plane.z # ; z = 'BR-aBR'
            ph = plane.phase; st = plane.stoich; cn = plane.config
            folder = '/'.join(plane.hop_path.split('/')[:-2])
            
            dist_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
            
            data2 = plane.hops.copy(deep=True); lat = ', all'
            if old_site == 'BR' : data2 = data2.query('old_is_BR == True');  lat = r', BR$\rightarrow$aBR'
            elif old_site == 'aBR' : data2 = data2.query('old_is_BR == False'); lat = r', aBR$\rightarrow$BR'
                
            ## if plotting shading, start min and max data structures
            mins = 50*np.ones(len(logbins)-1)
            maxes = np.zeros(len(logbins)-1)    
            
            # make a color map with a darker gray in the middle
            num_curves = min(est_max_radii[st],max_r_to_plot)+1
            # cmap = LinearSegmentedColormap('dark_cwb',segmentdata=hu.cdict, N=num_curves)
            # colors = [cmap(i) for i in np.linspace(0,1, num_curves)]
            colors = [batlow_even(i) for i in np.linspace(0,1, num_curves)]
            
            ## iterate marker for each set of data
            this_marker = next(markers)
            
            ## calculate the all-sites curve
            
                
            ## Counts / Log bin Onward & Reverse
            clb,_ = np.histogram(data2.new_resid_time[data2.rev_hop==True],bins=logbins)
            clo,_ = np.histogram(data2.new_resid_time[data2.rev_hop==False],bins=logbins)
            
            ## calculate the ratio of onward / (onward + reverse)
            ## then multiply by 1.5 for the correlation factor
            total_ratios = clo[clb>clb_cut]/(clb[clb>clb_cut]+clo[clb>clb_cut]) * 1.5
                
            if plot_gs : int_fun = interp1d(logcenters[clb>clb_cut], total_ratios, fill_value=(-1,1), bounds_error=False)
            
            if plot_total :
                ax.plot(logcenters[clb>clb_cut], total_ratios, c='k', alpha=0.5,
                        ms=4, label=f'{T1}K', linestyle='-',marker=this_marker)
            
            for r in range(num_curves) :
            
                ## select data
                data3 = data2.query(f'{dist_col} == @r')
                
                ## Counts / Log bin Onward & Reverse
                clb,_ = np.histogram(data3.new_resid_time[data3.rev_hop==True],bins=logbins)
                clo,_ = np.histogram(data3.new_resid_time[data3.rev_hop==False],bins=logbins)
                
                tag = f'{m} {T1}K, {z}' if full_label else f'{T1}K, {r}'
                
                ## make a special color for stoichiometric material
                if st == '100' : this_color = 'tab:green'
                else: this_color = colors[r]
                
                ## calculate the ratio of onward / (onward + reverse)
                ## then multiply by 1.5 for the correlation factor
                ratios = clo[clb>clb_cut]/(clb[clb>clb_cut]+clo[clb>clb_cut]) * 1.5
                
                ## update mins and maxes
                if shade_range and ratios.size > 0 :
                    # mins[(clb>clb_cut) & (mins[clb>clb_cut] > ratios)] = ratios[mins[clb>clb_cut] > ratios]
                    mins[clb>clb_cut] = np.minimum(mins[clb>clb_cut], ratios)
                    # maxes[(clb>clb_cut) & (maxes[clb>clb_cut] < ratios)] = ratios[maxes[clb>clb_cut] < ratios]
                    maxes[clb>clb_cut] = np.maximum(maxes[clb>clb_cut], ratios)
                ## Plot the ratio
                if not plot_total :
                    ax.plot(logcenters[clb>clb_cut], ratios, ms=4, label=tag,
                            linestyle='--',marker=this_marker, c=this_color, mec=this_color, mfc='none')
                    
            if shade_range : 
                ## Plot the shaded range
                mins[mins == 50] = np.nan
                maxes[maxes == 0] = np.nan
                
                ax.fill_between(logcenters, mins, maxes, alpha=0.4, facecolor=hu.metal_colors[m])
                
            ## add hop relaxation rates from van Hove function
            ## only reasonable if 
            if plot_gs :
                try: 
                    gs = hu.load_gs(folder+f'/{m}-*-gs-{T1}K*ps.csv', option='Funke', radii=rs_list)
                    ax.plot(gs, int_fun(gs), marker=this_marker, mfc='yellow', mec='k', zorder=3, ls='')
                except ValueError : print(f'could not do Gs decay for {m} {cn} {T1}. Check cell size.')
                    
            ## plot approximately the timescale when C_D goes to zero
            if plot_cdt :
                ylevel = 1; cd = -100; errs = [[10],[10]]
                if ph != 'beta' and T1 == 300 and m == 'K' : 
                    cd = 4.5e3; errs = [[1000],[1500]]
                elif ph != 'beta' and T1 == 600 and m == 'K' : 
                    cd = 25; errs = [[10],[130]]
                elif ph != 'beta' and T1 == 600 and m == 'Ag' : 
                    cd = 140; errs = [[110],[45]]
                elif ph != 'beta' and T1 == 600 and m == 'Na' : 
                    cd = 45; errs = [[35],[150]]
                elif ph == 'beta' and T1 == 600 and m == 'Ag' : 
                    cd = 1e4; errs = [[3e3],[3e3]]
                elif ph == 'beta' and T1 == 600 and m == 'Na' : 
                    cd = 2e4; errs = [[5e3],[5e3]]

                try : 
                    ax.errorbar(x=cd, y=int_fun(cd), xerr=errs, marker=this_marker, 
                                mfc='red', mec='k', zorder=3, ls='', c='red', ecolor='k')
                except : ax.errorbar(x=cd, y=ylevel, xerr=errs, marker=this_marker, 
                                mfc='red', mec='k', zorder=3, ls='', c='red', ecolor='k')
                
            color_counter += 1
            print(f'completed plane {m} {cn} {T1}')
            
    axtitle = m + (r' $\beta$' if ph == 'beta' else r' $\beta^{\prime\prime}$') + ('\n'+ distance_names[ph] if not plot_total else '')
    leg = ax.legend(title=f'{axtitle}', ncol = 2 if (not plot_total and len(these_planes) > 1) else 1, loc='lower right')
    plt.setp(leg.get_title(), multialignment='center')
    ax.plot([0,1e5], [1,1], ls=':', c='grey', lw=0.4)
    ax.set(xscale='log', yscale='linear', ylim=[0,1.05],xlim=(0.25,25e3),xlabel=r'Hop Residence Time $\tau$, ps')
    
axes[0].set(ylabel='Correlation Factor $f$'+lat, yticks=[0,0.2,0.4,0.6,0.8,1.0])
fig.tight_layout(pad=0.5)

# =============================================================================
# %% Figure S2 : load planes
# =============================================================================
    
planes_to_plot = all_planes.query('phase != "beta" & metal == "Na" & config in ["unsym_0", "symm_1"] & T1 in [230,300,473,600] & z == "016"')

planes_to_plot = planes_to_plot.sort_values(by=['T1','z','metal','stoich'])

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac, do_fill_times)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)

## labels based on defect placements
label_dict={'unsym_0':'not quenched', 'symm_1':'quenched'}
planes_data.config = [label_dict[x] for x in planes_data.config]

# =============================================================================
# %% Figure S2: by-site occupancy maps for individual planes & temperatures
# =============================================================================

# ========== automatic things below this line ==========

variable = 'T1'
var2 = 'config' 
var_values = set(planes_data[variable]) 
    
planes_data.sort_values(by=[var2, variable], inplace=True)

fig, axes = plt.subplots(len(planes_data)//len(var_values),len(var_values),
                         sharex=True, sharey=True, figsize=(12,6))

if len(planes_data) < 2 or len(axes) < 2: axes = [axes]
else : axes = axes.reshape(-1)

site_pts_list = list()
sites_list = list()
polys_list = list()
mins_list = list()
maxs_list = list()
cell_num_list = list()
boxes_list = list()
voros_list = list()

## iterate through planes to compose lists of quantities to be plotted 
for (i, plane), ax in zip(planes_data.iterrows(), axes):

    data = plane.hops; tt = plane.total_time; ph = plane.phase
    m = plane.metal; T1 = plane.T1; s = plane.stoich; z = plane.z 
    tt = plane.total_time
    BR_sites = plane.BR_sites
    sites_by_r = plane.sites_by_r
    
    ## dist_col is the dataframe column that has the distance index
    dist_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    
    if plane.num_planes > 1: 
        print('This block only works for single planes, not composite planes')
        continue
    
    ## to plot grid maps, I need: site_pts, 2D box, a freud voronoi to continue
    site_pts = np.copy(plane.site_pts)
    site_pts[:,-1] = 0 ## flatten in case it has not done so yet.
    box = freud.box.Box(Lx=plane.cell[0,0], Ly=plane.cell[1,1], is2D=True)
    vor = freud.locality.Voronoi(box)
    
    ## destroy z-coordinate here, and make polygons
    site_pts[:,-1] = 0
    vor.compute((box,site_pts))
                
    ## occupancies
    toplot_BR = pd.DataFrame(columns=['total',dist_col])
    toplot_aBR = pd.DataFrame(columns=['total',dist_col])
    for r in range(plane.max_r+1): ## range of distances to oxygens / counts of nearby defects
        total_BR, total_aBR = hu.site_occupancies(data,set(sites_by_r[r]),BR_sites,tt,plane,r=r)
        toplot_BR = toplot_BR.append(total_BR,sort=False)
        toplot_aBR = toplot_aBR.append(total_aBR,sort=False)
    sites = toplot_BR.append(toplot_aBR).total
        
    polys_all = [ vor.polytopes[int(i)] for i in list(sites.index) ]
    cell_nums = [ int(i)                for i in list(sites.index) ]
    
    ## save limits and polygons for plotting
    sites_list.append(sites)
    polys_list.append(polys_all)
    mins_list.append(min(sites))
    maxs_list.append(max(sites))
    cell_num_list.append(cell_nums)
    
    ## also save lists for freud tessellations
    site_pts_list.append(site_pts)
    boxes_list.append(box)
    voros_list.append(vor)
    
## create bounds for coloring below
# bounds = [min([x if not np.isnan(x) else 0 for x in mins_list]), max([x if not np.isnan(x) else 0 for x in maxs_list])]
bounds = [0,1]    

## this is the loop that actually creates the figure
for (i,plane), site_vals, polys, ax, site_pts, box, vor, cns \
    in zip(planes_data.iterrows(), sites_list, polys_list, axes, site_pts_list, 
           boxes_list, voros_list, cell_num_list) :
    
    plt.sca(ax)
    
    v = str(plane[variable]); v2 = ['', str(plane[var2])][var2 != 'phase']
    m = plane.metal;  T1 = plane.T1;  z = plane.z; ph = plane.phase
    
    Lx = plane.cell[0,0]; Ly = plane.cell[1,1]
    
    ## make up axes labels for the plot
    axt  = v + ['','K'][variable=='T1'] 
    axt2 = ', ' + ['','z='][var2=='z'] +v2+ ['','K'][var2 =='T1'] 
    
    ## plot the whole lattice, coloring by property chosen above
    hu.draw_voronoi(box, site_pts, polys, draw_box=True, color_by_property=site_vals, 
                    alpha=0.75, property_clim=bounds,cmap='YlOrRd') ## cell_numbers=cns
    ## set xy limits and the such
    ax.set(aspect=1,xlim=(-0.55*Lx,0.55*Lx),ylim=(-0.55*Ly,0.55*Ly),
         xticks=[], yticks=[], ylabel=axt2[2:])

    ## having pre-loaded the defect coordinates separately up top, plot them
    if ph != 'beta' or 'M' in cn : 
        spec = {'facecolors':'tab:green', 'edgecolors':'k'}
        half1 = plane.defects.query('-11 < z < -4')
        ax.scatter(half1.x, half1.y, s=20, alpha=1, **spec)
        half2 = plane.defects.query('-11 > z or z > -4')
        ax.scatter(half2.x, half2.y, s=40, alpha=0.5, **spec)
    else : 
        spec = {'facecolors':'tab:red', 'edgecolors':'k', 's':40}
        ax.scatter(plane.defects.x, plane.defects.y, **spec)

    ax.set(title=axt) ## +axt
    
## make labels: general 2D case
## make top, left, and bottom labels non-empty
for n, ax in enumerate(axes.flat):
    if n >= len(var_values): ## not top row
        ax.set(title='')
    if n % len(var_values) != 0: ## not left column
        ax.set(ylabel='')
    if len(axes.flat) - n > len(var_values) : ## not bottom row
        ax.set(xlabel='')
        
fig.tight_layout(pad=0.25)
    
# =============================================================================
# %% Figure 4 (+ Figure S6): pre-load data so it does not take time every time 
# =============================================================================
    
planes_to_plot = all_planes.query('phase != "beta" & num_planes > 2 & config == "unsym_0" & T1 == 600')
# planes_to_plot = all_planes.query('phase == "beta" & num_planes > 3 & config == "120_4" & T1 == 1000')

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac=False, do_fill_times=False)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)

# =============================================================================
# %% Figure 4b-d (+ Figure S6): block for 2D plot with squares (transition matrix)
# ## study hop characteristics by distances from oxygen defects
# ## 2020/04/07: for single planes site_pts_list, boxes_list, voros_list can be included
# ## these plots are resolved by from and to distances, i.e. these are scalars by edge
# =============================================================================

subject = 'Hops' ## Options: 'Hops' or 'Flybys' or '# Edges'
onw = True       ## if True, track onward hops only
abr = True       ## if True, track hops into aBR sites only (best with onward)
per_site = True  ## if True, normalize by the number of edges between sites

write_max = False ## write out the max value on the plot

## assign the variable over which to plot multiple panels
variable = 'metal'

# ========== automatic things below this line ==========

var2 = 'phase' ## set a default, this gets assigned automatically
var_values = set(planes_data[variable])

## if a second variable is varied, then find the second variable
relevant_vars = ['config', 'metal','T1','z','stoich','exclude']
relevant_vars.remove(variable)
if len(planes_data) > len(var_values): 
    for rv in relevant_vars: 
        if len(set(planes_data[rv])) > 1: var2 = rv; break

fig, axes = plt.subplots(len(planes_data)//len(var_values),len(var_values), 
                         sharey=True, sharex=True, figsize=(2+2*len(var_values),3.1))

## sort data so that it matches axes being cycled
planes_data.sort_values(by=[var2, variable], inplace=True)

if len(planes_data) < 2 or len(axes) < 2: axes = [axes]
else : axes = axes.reshape(-1)

## iterate to set common limits later
for ax, (i, plane) in zip(axes, planes_data.iterrows()):
    
    plt.sca(ax)
    
    m = plane.metal; T1 = plane.T1; z = plane.z; tt = plane.total_time/1000
    v = str(plane[variable]); v2 = [str(plane[var2]), ''][var2 == 'phase']    
    max_r = plane.max_r; ph = plane.phase
    
    ## columns with distances : new and old
    new_r_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    old_r_col = 'old_ox_r' if ph == 'beta' else 'old_mg_count'
    
    data = plane.hops.copy(deep=True).astype({old_r_col:'int32', new_r_col:'int32'}) 
    
    edge_distances = plane.edge_distances
        
    if subject == 'Flybys' : onw = True; data = data.query('new_resid_time < 1.1 & old_is_BR==False')
    
    ## apply the onwards flag
    if onw : data = data.query('rev_hop == False')
    if abr and ph == 'beta' : data = data.query('old_is_BR == True')
    
    ## check reasonableness and count using groupby
    # print max(data.new_ox_r.unique()), max(data.new_ox_r.unique()) 
    # cts = data.groupby([old_r_col,new_r_col]).size().reset_index()
    cts = data.groupby(['old_cell','new_cell']).agg({'ion':'size',old_r_col:'mean',new_r_col:'mean'}) \
              .reset_index(drop=True).groupby([old_r_col,new_r_col]).agg(['mean','std'])
    # print cts.max(), cts.sum()
    
    ## transform to np 2D array for dividing by edge counts and meshgrid
    cts_norm = np.zeros((int(data[old_r_col].max())+1, int(data[new_r_col].max())+1))
    cts_stds = np.zeros((int(data[old_r_col].max())+1, int(data[new_r_col].max())+1))
    for s1 in range(data[old_r_col].max()+1):
        for s2 in range(data[new_r_col].max()+1):
            try : 
                cts_norm[s1,s2] = cts.query(f'{old_r_col} == @s1 & {new_r_col} == @s2')[('ion', 'mean')] # /edge_distances[s1,s2]
                cts_stds[s1,s2] = cts.query(f'{old_r_col} == @s1 & {new_r_col} == @s2')[('ion',  'std')]
                # cts_norm[s1,s2] = cts.query(f'{old_r_col} == @s1 & {new_r_col} == @s2')[0]/edge_distances[s1,s2]
            except : cts_norm[s1,s2] = 0
    
    if subject == '# Edges': ## Plot connectivity
        ax.pcolor(np.arange(cts[old_r_col].max()+2)-0.5,
                  np.arange(cts[new_r_col].max()+2)-0.5,
                  edge_distances,cmap='viridis')
        maxval = str(int(edge_distances.max()))
        # print(edge_distances)
        
    elif subject == 'Hops' :
        if not per_site: ## just plot all the hops
            ax.hist2d(data[old_r_col],data[new_r_col],
                      bins=[max_r+1,max_r+1], range=[[-0.5,max_r+0.5],[-0.5,max_r+0.5]])
            maxval=f'{cts[0].max()/tt:.2f} /ns'
        else: ## normalize by # edges    
            ax.pcolor(np.arange(data[old_r_col].max()+2)-0.5,
                      np.arange(data[new_r_col].max()+2)-0.5,cts_norm.T,cmap='viridis')
            maxval = f'{np.max(cts_norm)/tt:.1f}±{cts_stds[np.where(cts_norm == np.max(cts_norm))][0]/tt:.1f} /ns'
            # print(cts_norm.round(1)/tt, cts_stds.round(1), tt)
    elif subject == 'Flybys' :
        if not per_site: ## just plot all the hops
            ax.hist2d(data[old_r_col],data[new_r_col],bins=[max_r+1,max_r+1], 
                      range=[[-0.5,max_r+0.5],[-0.5,max_r+0.5]])
            maxval=f'{cts[0].max()/tt:.1f} /ns'
        else: ## normalize by # edges    
            ax.pcolor(np.arange(cts[old_r_col].max()+2)-0.5,
                      np.arange(cts[new_r_col].max()+2)-0.5,
                      cts_norm.T,cmap='viridis')
            maxval = f'{cts_norm.max()/tt:.2f} /ns'
    
    ## print matrices
    print(f'=== {v} {v2}, {tt:.1f} ns ===')
    print(cts_norm.round(1)/tt)
    print(cts_stds.round(1)/tt)
    
    ## make up titles and stuff
    axt  = ', ' +v+ ['','K'][variable=='T1'] 
    axt2 = ', ' +v2+ ['','K'][var2 =='T1'] 
    
    # ax.set(title=axt[2:])
    ax.set(xlabel=r'r$_{from}$', aspect=1, 
           ylabel=r'r$_{to}$'+['',axt2][var2 is not None])
    ticklist = list(range(0,max_r+1,2))
    ax.set_yticks(ticklist); ax.set_xticks(ticklist)
        
    ## label the plot with the max value
    if write_max :
        if subject ==  '# Edges':
            ax.text(-0.25,max_r-1.125,f'max={maxval}',fontsize=10,color='tab:orange')
        elif onw : 
            ax.text(-0.25,-0.125,f'max={maxval}',fontsize=10,color='tab:orange')
        else : 
            ax.text(-0.25,max_r-1.125,f'max={maxval}',fontsize=10,color='tab:orange')
            
    ## write the x- and y- axes labels
    if ph == 'beta' :
        ax.set_xlabel(r'# Sites to $O_i^{\prime\prime}$, hop origin', fontsize=13.5)
        ax.set_ylabel(r'# Sites to $O_i^{\prime\prime}$, destination', fontsize=13.5)
        # ax.set(la)
    else :
        ax.set(xlabel = r'# Mg$_{Al}^\prime$ at hop origin', 
               ylabel = r'# Mg$_{Al}^\prime$ at destination')
    
    del data


## make labels: general 2D case
## make top, left, and bottom labels non-empty
if len(axes) > 1:
    for n, ax in enumerate(axes.flat):
        if n >= len(var_values): ## not top row
            ax.set(title='')
        if n % len(var_values) != 0: ## not left column
            ax.set(ylabel='')
        if len(axes.flat) - n > len(var_values) : ## not bottom row
            ax.set(xlabel='')
        
for i in range(5) : fig.tight_layout(pad=0.5, w_pad=0.25)
    
# =============================================================================
# %% Figure 4e-g, Figure S6e-g: Calc site energies from by-site occupancy
# ## here, non-hops automatically are 1, and empties automatically are 0
# =============================================================================

variable = 'metal'  ## << do pick this
verbose = True  ## flag for how much print() outputs to show. False is less.

plot_swarm = False  ## for reducing clutter, skip plotting the swarm
plot_box = True    ## for reducing clutter

# ========== automatic things below this line ==========

var_values = set(planes_data[variable])

fig, axes = plt.subplots(1,len(var_values),
                          sharex=True, sharey=True,figsize=(6*len(var_values),4.8 if not plot_swarm else 6))
if len(var_values) == 1 : axes = [axes]

gro = [] ## list of average triples: (degeneracy g, distance r, occupancy o)
site_gro = pd.DataFrame(columns=['o','r']) ## site-specific list of triples: (degeneracy g=1, distance r, occupancy o)

for var, ax in zip(sorted(var_values), axes):
    
    Ts = []
    energies = [[] for r in range(20)]
    occs = [[] for r in range(20)]
    
    for i, plane in planes_data.iterrows():
        if plane[variable] == var:
            
            gro = [] ## reset for the next plane
            site_gro = pd.DataFrame(columns=['o','r']) ## reset for the next plane
            
            mm = plane.metal; T1 = plane.T1; z = plane.z; ph = plane.phase
            max_r = plane.max_r; tt = plane.total_time; cn = plane.config; st = plane.stoich
            
            new_r_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
            
            data2 = plane.hops
            BR_sites = plane.BR_sites
            sites_by_r = plane.sites_by_r
                        
            Ts.append(T1) ## for later plotting
        
            for r in range(min(max_r+1,20)) : ## for every distance
                
                total_BR, total_aBR = hu.site_occupancies(data2,set(sites_by_r[r]),
                                                          BR_sites,tt,plane,r=r,verbose=False)
                num_BR = len(total_BR); avg_BR = total_BR.total.mean()
                num_aBR = len(total_aBR); avg_aBR = total_aBR.total.mean()
                                
                # quick display of how much the defect is dissociated
                avg_r = (total_BR.total.sum() + total_aBR.total.sum()) / (num_BR + num_aBR) * 2
                print(f'\n{m:2s} {cn} {T1:4d}K, r={r}, would-be stoichiometry: {avg_r:.2f}')
                
                try:
                    
                    ## write occupancies for every-site energies
                    gro.append( (num_BR,  r, avg_BR , 'B') ) 
                    gro.append( (num_aBR, r, avg_aBR, 'A') )
                    
                    total = total_BR.append(total_aBR).rename(columns={'total':'o', new_r_col:'r'})
                    site_gro = site_gro.append(total)
                    
                    ## use nondegenerate method; you'd think num_BR==num_aBR, but NO.
                    EaBR = hu.two_state_nondeg_energy(num_BR, num_aBR,
                                                      num_BR*avg_BR, num_aBR*avg_aBR, T1, lb=-2)
                    EBR  = hu.two_state_nondeg_energy(num_aBR, num_BR,
                                                      num_aBR*avg_aBR, num_BR*avg_BR, T1, lb=-2)
                    
                    if verbose:
                        
                        max_BR = total_BR.total.max(); max_aBR = total_aBR.total.max()
                        min_BR = total_BR.total.min(); min_aBR = total_aBR.total.min()
                        
                        print(f'  {m:2s} {T1:4d}K, r={r}, {len(set(sites_by_r[r])):3d} sites ({num_BR} BR and {num_aBR} a-BR):')
                        print(f'  BR occup {avg_BR*100. :.2f}% ({min_BR*100. :.2f}-{max_BR*100. :.2f}), E_aBR {-EBR:.3f}eV')
                        print(f'a-BR occup {avg_aBR*100.:.2f}%, ({min_aBR*100. :.2f}-{max_aBR*100. :.2f}), E_aBR {EaBR:.3f}eV')
                    
                    energies[r].append(np.mean([EaBR,-EBR]))
                    
                except (ValueError, AssertionError, IndexError) as error:
                    print(f'{m:2s} {T1:4d}K, r={r}, {len(set(sites_by_r[r])):3d} sites ({num_BR} BR and {num_aBR} a-BR):')
                    print('error:', error)
                    
                    energies[r].append(np.nan)
                    # occs[r].append(occ_aBR) 
            
            ## after all distances are done, compute full 10-site energies
            if T1 > 200 and ( ph != 'beta') or (ph == 'beta' and hu.s2n(st) > 100) :
                
                ## (binning by distance to defects only works with Oi defects)
                if ph == 'beta' and 'M' not in plane.exclude:
                    df = hu.multi_state_energies(gro, T1)
                    
                    if ph != 'beta' : df.site = ['direct' if x == 'B' else 'offset' for x in df.site]
                    else : df.site = ['BR' if x == 'B' else 'aBR' for x in df.site] 
                    df.es -= df.es.min()
                    df.sort_values(by=['site'])
                
                    ## Plot one average point per distance
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(data=df, x='rs', y='es', style='site', ax=ax2, s=10) ## for the legend
                    ax2.scatter(df.query('site in ["BR","direct"]').rs, df.query('site in ["BR","direct"]').es, s=100, marker='o', c=hu.metal_colors[m])
                    ax2.scatter(df.query('site in ["aBR", "offset"]').rs, df.query('site in ["aBR", "offset"]').es, s=100, marker='X', c=hu.metal_colors[m])
                    ax2.set(title=f'{m} {T1}K', xlim=[min(df.rs)-0.25, max(df.rs)+0.25])
                    ax2.set(xlabel=distance_names[ph], ylabel='Site Energy, eV')
                    ax2.set_xticks(range(5))
                    fig2.tight_layout()
                
                ## calculate energies for every site
                site_gro['g'] = 1
                site_gro = hu.multi_site_es(site_gro,T1)
                print('{:2s} {}K site energies: {:.3f}-{:.3f} eV'.format(m,T1,site_gro.e.min(), site_gro.e.max()))
                if ph != 'beta' : site_gro.site = ['direct' if x == 'BR' else 'offset' for x in site_gro.site]
                
                ## plot box + swarm plots. Originally 1 figure per plane
                # fig3, ax3 = plt.subplots(figsize=(9,8))
                # sns.boxplot(ax=ax3, x='r', y='e', data=site_gro, hue='site', fliersize=0, sym='', palette='pastel')
                # sns.swarmplot(ax=ax3, x='r', y='e', data=site_gro, hue='site', alpha=0.75, size=3)
                # ax3.set(ylabel='Energy, eV', xlabel=distance_names[ph], title=f'{m} {T1}K')
                # fig3.tight_layout()
                
                if plot_box :
                    sns.boxplot(ax=ax, x='r', y='e', data=site_gro.sort_values(by='site'), 
                                hue='site', fliersize=0, sym='', palette='pastel')
                if plot_swarm :
                    sns.swarmplot(ax=ax, x='r', y='e', data=site_gro.sort_values(by='site'), 
                                  hue='site', alpha=0.75, size=3)
                ax.set(xlabel=distance_names[ph], ylabel='')
                # ax.set(title=f'{mm} {T1}K')
                
        ax.legend(title=f'{mm} {phases[ph]} site:', fontsize=12, title_fontsize=12,
                  loc='lower left' if ph != 'beta' else 'upper center')

axes[0].set(ylabel='Relative Energy, eV')
fig.tight_layout(pad=0.5, w_pad=0.25)  

# =============================================================================
# %% Figures S4-5: plot smoothed probability distribution functions (PDF's) 
# =============================================================================

subject = 'PDF' ## 'PDF', 'CDF', 'fill'
variable = 'metal' ## Pick the 1st variable to plot << DO THIS

BR_disagg = False ## plot separate things for BR/aBR; assumes 'old_is_BR' is an index level
onw_disagg = False ## implemented, but not at the same time as BR/aBR
plot_onw = False ## (for BR-aBR) onward hops are plotted if true, back-hops if false. Fill: new ion

dist = 'new' ## 'new' or 'old' refer to new_cell/old_cell; plotting hops 'into' a distance uses 'new'
one_r = None ## if not None, then keep one value of 'dist' above, and disaggregate by a second distance
# r_to_plot = [0,1,2,3,4,5,6] ## if one_r is None, plot distances in this iterable rather than everything
r_to_plot = [] ## if one_r is None, plot distances in this iterable rather than everything
plot_total = True  ## plot the full, all-distances distributions

per_site = False    ## boolean flag; if True, then divide # of hops by # sites
downsample = True   ## smooth with a gaussian window
hl = 200            ## [fs] sigma for a gaussian window

append_T = False ## cosmetic : temperature will be appended to plot labels
long_time_limit = 2000 ## [ps] upper limit on time for which to plot totals

guides = True   ## plot power law guides 
rs_list = [[0.01,1.7], [0.01,4.6]] ## for Gs markers
guide_v2 = 300    ## value of var2 at which to plot guidelines, typically temp

# ========== automatic things below this line ==========

## no need to smooth out CDF's
if subject == 'CDF' : downsample=False

var2 = None     ## this gets determined later; leave as None

## get values that are varied in the current set of planes. 
var_values = set(planes_data[variable])

## automatically find the second variable
relevant_vars = ['metal','T1','z','stoich','exclude', 'config']
relevant_vars.remove(variable)
if len(planes_data) > len(var_values): ## a second variable is varied
    for rv in relevant_vars: 
        if len(set(planes_data[rv])) > 1: var2 = rv; break
if variable != 'T1' and var2 != 'T1': append_T = True

## sort planes by the two variables for easy plotting and switching between axes
if var2 is not None: planes_data = planes_data.sort_values(by=[variable, var2], axis=0)
else: planes_data = planes_data.sort_values(by=variable, axis=0)

## set up plots in the right dimensions 
num_plots = max(1,2*BR_disagg+2*onw_disagg)
fig, axes = plt.subplots(num_plots, len(var_values), sharex=True, 
                         figsize=(4.5*len(var_values),4.5*num_plots))

if not isinstance(axes, np.ndarray) : axes = [axes]
else : axes = axes.reshape(-1)

## Set up a cycler through axes, and a debugging counter.
## This helps to plot the right curves in the right axes
ax_iter = cycle(axes) 
aa_iter = cycle(list(range(len(axes))))

## set up shared y scales by row of plots (all x scales are shared)
## for some reason only works upfront here, not after plotting
for n in range(num_plots) :
    for i, ax in enumerate(axes[n*len(var_values)+1:(n+1)*len(var_values)]):
        axes[n*len(var_values)]._shared_y_axes.join(ax,axes[n*len(var_values)])
   
## iterate over planes 
for i, plane in planes_data.reset_index().iterrows():
    
    ## keep track of how many variables have switched between planes
    ## this helps to cycle to the correct axes later 
    switch_count = 0
    if i != 0:
        if v != str(plane[variable]) : switch_count += 1
        if var2 is not None:
            switch_count -= 1 if v2 == str(plane[var2]) else -1
        else : switch_count += 1
    else: ax = next(ax_iter)
    
    ## make shorthands for metadata
    mm = plane.metal; T1 = plane.T1; z = plane.z; data = plane.hops.copy(deep=True) #.round({'new_resid_time':3})
    v = str(plane[variable]); v2 = str(plane[var2]) if var2 is not None else 'no var2'
    st = plane.stoich; ph = plane.phase; cn = plane.config
    site_counts = [x.size for x in plane.sites_by_r]
    folder = '/'.join(plane.hop_path.split('/')[:-2])
    
    ## round time points above
    
    ## columns with distances : new and old
    new_r_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    old_r_col = 'old_ox_r' if ph == 'beta' else 'old_mg_count'
    
    ## BR site column, reverse/onward column
    BR_site_col = 'site_is_BR' if subject == 'fill' else 'old_is_BR'
    rev_col = 'refill' if subject == 'fill' else 'rev_hop'
    
    ## adjust for the defect-free stoichiometries so they do not get skipped
    ## This is automatically beta
    if cn == '100_0' and one_r is not None:
            data.old_ox_r = one_r
            data.new_ox_r = one_r
            
    if cn == '100_0' : 
        site_counts = np.zeros(100).astype(int)
        site_counts[-1] = len(plane.site_pts) * len(plane.site_pts[0])
            
    if subject == 'fill' :
        data = plane.fill_times.copy(deep=True)
        # data.r_to_defect = one_r
        new_r_col = 'r_to_defect'
        old_r_col = 'r_to_defect'
            
    ## make a set of sub-divided DF's; then apply a map to them depending on options
    data_aBR = data.query(f'{BR_site_col} == False')
    data_BR =  data.query(f'{BR_site_col} ==  True')
    data_onw = data.query(f'{rev_col} == False') ## keep
    data_rev = data.query(f'{rev_col} ==  True') ## keep
    if plot_onw:
        data_aBR = data_aBR.query(f'{rev_col} == False')
        data_BR  =  data_BR.query(f'{rev_col} == False')
    
    ## labels for groupby if subject == PDF
    r1 = new_r_col if dist == 'new' else old_r_col
    r2 = new_r_col if dist != 'new' else old_r_col
    
    ## make a color map with a darker gray in the middle
    max_r = int(max( data[r1].max(), data[r2].max() ))
    cmap = LinearSegmentedColormap('dark_cwb',segmentdata=hu.cdict,N=max_r+1)
    colors = [cmap(i) for i in np.linspace(0,1,max_r+1)]
    
    ## make up titles for axes: first variable on top
    axt  = ', ' +v + ['','K'][variable=='T1']
    axt2 = ', ' + (v2 if var2 is not None else f'{T1}K' )+ ['','K'][var2 =='T1'] 
    
    if one_r is not None : axt += ', '+['to', 'from'][dist=='old'] + f' r={one_r}'
        
    ## Plot PDF's. 
    ## Possible confusion with earlier work: 'BR' meant from-BR. Here: to-BR. 
    ## This is addressed by explicit labels on the left y-axis
    fun = lambda df: pd.DataFrame.groupby(df, [r1, 'new_resid_time']).new_resid_time \
            .agg('count').pipe(pd.DataFrame).rename(columns={'new_resid_time':'frequency'}) \
        
    
    if one_r is not None :
        fun = lambda df: pd.DataFrame.groupby(df, [r1, r2,'new_resid_time']).new_resid_time \
                        .agg('count').pipe(pd.DataFrame).rename(columns={'new_resid_time':'frequency'}) \
                        .query(r1+' == @one_r').reset_index(level=0,drop=True)
                        
    if subject == 'fill' :
        fun = lambda df : pd.DataFrame.groupby(df, [r1, 'time']).time.agg('count') \
                         .pipe(pd.DataFrame).rename(columns={'time':'frequency'})
                         
        if one_r is not None :
            fun = lambda df : df.groupby([r1, 'time']).time.agg('count').pipe(pd.DataFrame) \
                             .rename(columns={'time':'frequency'}).query(r1+' == @one_r')
         
    toplot, toplot_BR, toplot_aBR, toplot_onw, toplot_rev = [fun(x) for x in [data, data_BR, data_aBR, data_onw, data_rev]]        
            
    ## trying to manage memory
    del data, data_BR, data_aBR, data_onw, data_rev
        
    print(f'plotting {v} and {v2}')
    
    ## if an extra variable changes between this plane & last, cycle axes again
    for j in range(switch_count // 2) : 
        ax = next(ax_iter)
        a  = next(aa_iter)
    
    if BR_disagg:
        
        ## plot the total distribution for to-BR, and Gs
        if plot_total :
            this = toplot_BR.reset_index(level=0,drop=True).groupby('new_resid_time' if subject != 'fill' else 'time').agg('sum')
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            else: this = this.query(f'index < {long_time_limit}') / (sum(site_counts) if per_site else 1)
            if downsample : this = hu.pdf_smooth(this, hl) / plane.total_time * 1e4
            ax.plot(this.index, this.frequency, label=axt2[2:]+' all', c='g' if st == '100' else 'k')
            
            try :
                interp_fun = interp1d(this.index, this.frequency)
                gs = None
                gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=[[0.01,1.7], [0.01,4.4]])
                gs = [x for x in gs if x <= this.index.max()]
                ax.plot(gs, interp_fun(gs), marker='o', mfc='yellow', mec='k', zorder=4, ls='', markersize=4)
            except: pass
            
        
        ## plot the by-radius distribution for to-BR
        for r in toplot_BR.index.get_level_values([r1,r2][one_r is not None]).unique().astype(int) :
            if st == '100' : continue
            if r not in r_to_plot : continue
            this = toplot_BR.query(f'{[r1,r2][one_r is not None]} == @r').reset_index(level=0,drop=True)
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            elif per_site : this.frequency /= site_counts[r]
            if downsample : this = hu.pdf_smooth(this.query(f'index < {long_time_limit}'), hl)
            ax.plot(this.index, this.frequency, label=axt2[2:]+' r='+str(r),
                    c=[colors[r], hu.metal_colors[m]][var2=='metal'])
            
        if guides and v2 == str(guide_v2) : 
            if mm == 'Ag' and cn == '120_4' :
                ax.plot([6,60],[30,0.3], c='k', lw=0.4)
            elif mm == 'K' and cn == '120_4' :
                ax.plot([1.5, 15],[4,0.4], c='k', lw=0.4)
                ax.plot([10,100],[160,1.6], c='k', lw=0.4)
            elif mm == 'Na' and cn == '120_4' :
                ax.plot([1.2,7.2],[300,50], c='k', lw=0.4)
                ax.plot([15,75],[200,8], c='k', lw=0.4)
            
        ## fluff
        # ax.set(title=subject+axt, ylabel = r'BR$\rightarrow$aBR' +[' & back',' onward'][plot_onw])
        if subject == 'fill' : ax.set(ylabel='BR ' + ('new ion' if plot_onw else 'returning'))
        
        ax.set(ylabel = r'BR$\rightarrow$aBR' +[' Attempts',' Diffusion Events'][plot_onw] + ' / 10 ns')
        ax.legend(title = f'{mm} {phases[ph]}')
        ## scroll through axes
        for j in range(len(var_values)): 
            ax = next(ax_iter)
            a  = next(aa_iter)
        # print(f'aBR axes: {a % (len(var_values) * num_plots)} \n')
        
        ## plot the total for to-aBR, and Gs
        if plot_total :
            this = toplot_aBR.reset_index(level=0,drop=True).groupby('new_resid_time' if subject != 'fill' else 'time').agg('sum')
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            else: this = this.query(f'index < {long_time_limit}') / (sum(site_counts) if per_site else 1)
            if downsample : this = hu.pdf_smooth(this, hl) / plane.total_time * 1e4
            ax.plot(this.index, this.frequency, label=axt2[2:]+' all',c='g' if st == '100' else 'k')
            
            try :
                interp_fun = interp1d(this.index, this.frequency)
                gs = None
                gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=[[0.01,1.7], [0.01,4.4]])
                gs = [x for x in gs if x <= this.index.max()]
                ax.plot(gs, interp_fun(gs), marker='o', mfc='yellow', mec='k', zorder=4, ls='', markersize=4)
            except: pass
        
        ## plot the by-radius for to-aBR
        for r in toplot_aBR.index.get_level_values([r1,r2][one_r is not None]).unique().astype(int) :
            if st == '100' : continue
            if r not in r_to_plot : continue
            this = toplot_aBR.query(f'{[r1,r2][one_r is not None]} == @r').reset_index(level=0,drop=True)
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            elif per_site : this.frequency /= site_counts[r]
            if downsample : this = hu.pdf_smooth(this.query(f'index < {long_time_limit}'), hl)
            ax.plot(this.index, this.frequency, label=axt2[2:]+' r='+str(r),
                    c=[colors[r], hu.metal_colors[mm]][var2=='metal'])
            
        if guides and v2 == str(guide_v2) : 
            if mm == 'Ag' and cn == '120_4' :
                ax.plot([3,10],[20,6], c='k', lw=0.4)
                ax.plot([40,200],[200,8], c='k', lw=0.4)
            elif mm == 'K' and cn == '120_4' :
                ax.plot([40,200],[50,2], c='k', lw=0.4)
                ax.plot([1.2,6],[2,0.4], c='k', lw=0.4)
            elif mm == 'Na' and cn == '120_4' :
                ax.plot([2,10],[20,4], c='k', lw=0.4)
                ax.plot([50,200],[100,6.25], c='k', lw=0.4)
            
        ## fluff
        # ax.set(title=subject+axt, ylabel = r'aBR$\rightarrow$BR'+[' & back',' onward'][plot_onw] + ['', f', {T1}K'][append_T])
        if subject == 'fill' : ax.set(ylabel='aBR ' + ('new ion' if plot_onw else 'returning'))
        ax.set(ylabel = r'aBR$\rightarrow$BR' +[' Attempts',' Diffusion Events'][plot_onw] + ' / 10 ns')
        ax.legend(title = f'{mm} {phases[ph]}')
        ## scroll through axes for the next plane (debugging commented out)
        for j in range(len(var_values)): 
            ax = next(ax_iter)
            a  = next(aa_iter)
        # print(f'next axes: {a % (len(var_values) * num_plots)}')
        # print('going to next plane')

    elif onw_disagg: 
        
        # print(f'onw axes: {a % (len(var_values) * num_plots)} ({a})')
        ## should be similar to BR_disagg
        ## plot the total distribution for onw
        if plot_total:
            this = toplot_onw.reset_index(level=0,drop=True).groupby('new_resid_time' if subject != 'fill' else 'time').agg('sum')
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            else: this = this.query(f'index < {long_time_limit}') / (sum(site_counts) if per_site else 1)
            if downsample : this = hu.pdf_smooth(this, hl) / plane.total_time * 1e4
            ax.plot(this.index, this.frequency, label=axt2[2:]+' all', c='g' if hu.s2n(str(st)) == 100 else 'k')
            
            try :
                interp_fun = interp1d(this.index, this.frequency)
                gs = None
                gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=[[0.01,1.7], [0.01,4.4]])
                gs = [x for x in gs if x <= this.index.max()]
                ax.plot(gs, interp_fun(gs), marker='o', mfc='yellow', mec='k', zorder=4, ls='')
            except: pass
        
        ## plot the by-radius distribution for to-BR
        for r in toplot_onw.index.get_level_values([r1,r2][one_r is not None]).unique().astype(int) :
            if st == '100' : continue
            if r not in r_to_plot : continue
            this = toplot_onw.query(f'{[r1,r2][one_r is not None]} == @r').reset_index(level=0,drop=True)
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            elif per_site : this.frequency /= site_counts[r]
            if downsample : this = hu.pdf_smooth(this.query(f'index < {long_time_limit}'), hl)
            ax.plot(this.index, this.frequency, label=axt2[2:]+' '+str(r),
                    c=[colors[r], hu.metal_colors[mm]][var2=='metal'])
            
        ## fluff
        ax.set(title=subject+axt, ylabel = 'Onward' + (f', {T1}K' if append_T else ''))
        if subject == 'fill' : ax.set(ylabel='New ion ' + (f', {T1}K' if append_T else ''))
        ## scroll through axes
        for j in range(len(var_values)): 
            ax = next(ax_iter)
            a  = next(aa_iter)
        # print(f'rev axes: {a % (len(var_values) * num_plots)} ({a})\n')
        
        ## plot the total for to-aBR
        this = toplot_rev.reset_index(level=0,drop=True).groupby('new_resid_time' if subject != 'fill' else 'time').agg('sum')
        if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
        else: this = this.query(f'index < {long_time_limit}') / (sum(site_counts) if per_site else 1) 
        if plot_total:
            if downsample : this = hu.pdf_smooth(this, hl) / plane.total_time * 1e4
            ax.plot(this.index, this.frequency, label=axt2[2:]+' all',c='g' if hu.s2n(str(st)) == 100 else 'k')
            
            try :
                interp_fun = interp1d(this.index, this.frequency)
                gs = None
                gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=[[0.01,1.7], [0.01,4.4]])
                gs = [x for x in gs if x <= this.index.max()]
                ax.plot(gs, interp_fun(gs), marker='o', mfc='yellow', mec='k', zorder=4, ls='')
            except: pass
        
        ## plot the by-radius for to-aBR
        for r in toplot_rev.index.get_level_values([r1,r2][one_r is not None]).unique() :
            if st == '100' : continue
            if r not in r_to_plot : continue
            r = int(r)
            this = toplot_rev.query(f'{[r1,r2][one_r is not None]} == @r').reset_index(level=0,drop=True)
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            elif per_site : this.frequency /= site_counts[r]
            if downsample : this = hu.pdf_smooth(this.query(f'index < {long_time_limit}'), hl)
            ax.plot(this.index, this.frequency, label=axt2[2:]+' '+str(r),
                    c=[colors[r], hu.metal_colors[mm]][var2=='metal'])
            
        ## fluff
        ax.set(title=subject+axt, ylabel = r'Reverse' + ['', f', {T1}K'][append_T])
        if subject == 'fill' : ax.set(ylabel='Returning ion ' + (f', {T1}K' if append_T else ''))
        ## scroll through axes for the next plane (debugging commented out)
        for j in range(len(var_values)): 
            ax = next(ax_iter)
            a  = next(aa_iter)
        # print(f'next axes: {a % (len(var_values) * num_plots)}')
        # print('going to next plane')
        
    else : ## plot all/onward hops together. Useful for CDF & fitting
        
        ## if beta, disregard the hops at the defects when plotting everything
        if ph == 'beta' :
            toplot = toplot.loc[(2,0.00):]
            toplot_onw = toplot_onw.loc[(2,0.00):]
        ## plot the total distribution of all hops
        this = (toplot_onw if plot_onw else toplot).reset_index(level=0,drop=True) \
                .groupby('new_resid_time' if subject != 'fill' else 'time').agg('sum')
        if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
        else: this = this.query(f'index < {long_time_limit}')
        if downsample : 
            this = hu.pdf_smooth(this, hl).dropna() / plane.total_time * 1e4
            if this.empty : continue
        if plot_total:
            ax.plot(this.index, this.frequency, label=axt2[2:], c='g' if st == '100' else (hu.metal_colors[mm] if var2 == 'metal' else 'k'))
            
            try :
                gs = None
                gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=rs_list)
            except : 
                try : 
                    gs = hu.load_gs(folder+f'/{mm}-*-gs-{T1}K*ps.csv', option='Funke', radii=[[0.01,1.7]])
                except : pass
            interp_fun = interp1d(this.index, this.frequency)
            gs = [x for x in gs if x <= this.index.max()]
            ax.scatter(gs, interp_fun(gs), marker='o', s=16, edgecolors='k', facecolors='yellow', zorder=4)
            
            ## Plot power-law guidelines
            if guides and v2 == str(guide_v2) : 
                if not plot_onw: 
                    if mm == 'Ag' and ph == 'bdp' :
                        ax.plot([1.5,9],[300,50], c='k', lw=0.4)
                        ax.plot([100,300],[30/4,30/36], c='k', lw=0.4)
                    elif mm == 'K' and ph == 'bdp' :
                        ax.plot([2,10],[5e3, 1e3], c='k', lw=0.4)
                        ax.plot([40,300],[1e2, 100/7.5**2], c='k', lw=0.4)
                    elif mm == 'Na' and ph == 'bdp' :
                        ax.plot([2,10],[200, 40], c='k', lw=0.4)
                        ax.plot([60,300],[25, 1], c='k', lw=0.4)
                else : 
                    if mm == 'Ag' and ph == 'bdp' :
                        ax.plot([30,90],[0.75, 0.75/3], c='k', lw=0.4)
                        ax.plot([20,100],[250, 10], c='k', lw=0.4)
                    elif mm == 'K' and ph == 'bdp' :
                        ax.plot([50,100],[0.6, 0.3], c='k', lw=0.4)
                        ax.plot([40,300],[1e2, 100/7.5**2], c='k', lw=0.4)
                    elif mm == 'Na' and ph == 'bdp' :
                        ax.plot([20,40],[0.6, 0.3], c='k', lw=0.4)
                        ax.plot([100,200],[2,0.5], c='k', lw=0.4)
                        ax.plot([10, 30],[1800, 200], c='k', lw=0.4)
                    
        ## plot the by-radius distribution for all hops
        for r in toplot.index.get_level_values([r1,r2][one_r is not None]).unique() :
            if st == '100' : continue
            if r not in r_to_plot : continue
            r = int(r)
            this = toplot.query(f'{[r1,r2][one_r is not None]} == @r').reset_index(level=0,drop=True)
            if subject == 'CDF' : this.frequency = np.cumsum(this.frequency)/this.frequency.sum()
            elif downsample : this = hu.pdf_smooth(this, hl)
            ax.plot(this.index, this.frequency, label=axt2[2:]+' '+str(r),
                    c=[colors[r], hu.metal_colors[mm]][var2=='metal'])
            
        ## fluff
        # ax.set(title=subject+axt, ylabel = 'All Hops' + (f', {T1}K' if append_T else ''))
        ax.set(ylabel = 'Hopping Attempts' + (' per 10 ns' if subject == 'PDF' else ''))
        ax.legend(title = f'{mm} {phases[ph]}')
        ## scroll through axes
        for j in range(len(var_values)): 
            ax = next(ax_iter)
            a  = next(aa_iter)
        
## Plot guidelines or other subject-specific stuff
for ax in axes:    
    
    ## if plots of totals are the only plots, or there is only one r, re-color
    
    non_guide_lines = [x for x in ax.lines if x.get_label()[0] != '_']
    colors = [batlow_even(j) for j in np.linspace(0, 1, len(non_guide_lines))]
        
    if plot_total and not r_to_plot :
        for i, l in enumerate(non_guide_lines) :
            if ('ps' not in l.get_label()) and (var2 != 'metal') : l.set(color=colors[i])
            
    elif not plot_total and len(r_to_plot) == 1:
        for i, l in enumerate(non_guide_lines) : l.set(color=colors[i])
            
    ## scales and limits
    if subject == 'PDF' :
        ax.set(yscale='log', xscale='log', xlim=[0.05,50], xlabel=r'Residence Time $\tau$, ps')
        
    elif subject == 'CDF' :
        ax.set(xscale='log', xlim=[0.08,20], ylim=[0.1,0.992], xlabel='Residence Time $\tau$, ps')
        
        ax.set_yscale('logit') # , use_overline=True
        plt.setp(ax.get_yminorticklabels(), visible=False)
        
        axes[0].set(yticks=[0.10,0.50,0.90,0.99], yticklabels=['0.10','0.50','0.90','0.99'])
        
    elif subject == 'fill' :
        ax.set(yscale='log', xscale='log',xlim=[0.05,50], xlabel='Time to Fill, ps')
        
    ## remake legend with same title - but possibly new colors
    ax.legend(title=ax.get_legend().get_title().get_text())
        
## plotting fluff on top, left, and bottom
for n, ax in enumerate(axes):
    if n >= len(var_values): ## not top row
        ax.set(title='')
    if n % len(var_values) != 0: ## not left column
        ax.set(ylabel='')
        ax.set_yticks([])
    if len(axes) - n > len(var_values) : ## not bottom row
        ax.set(xlabel='')
for i in range(5) : fig.tight_layout(pad=0.5, w_pad=0.1)

# =============================================================================
# %% Figure S7: Trapping times for beta: load planes
# =============================================================================

planes_to_plot = all_planes.query('num_planes > 3 & config == "120_4" & T1 in [300,600,1000]') 

planes_to_plot = planes_to_plot.sort_values(by=['T1','z','metal','stoich'])

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac, do_fill_times)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)

# =============================================================================
# %% Trapping Times for beta: Figure S7
# =============================================================================

variable = 'metal'  ## << do pick this
verbose = True  ## flag for how much print() outputs to show. False is less.

logbins = np.logspace(-0.7,5.3,121)  ## sigma np.round(np.logspace(-0.7,4.3),2)

guides = True

# ========== automatic things below this line ==========

logcenters = np.sqrt(logbins[1:]*logbins[:-1])

var_values = set(planes_data[variable])

fig, axes = plt.subplots(1,len(var_values), sharex=True, sharey=True,
                          figsize=(4.5*len(var_values), 3.6))
if len(var_values) == 1 : axes = [axes]

for var, ax in zip(sorted(var_values), axes):
    
    for i, plane in planes_data.iterrows():
        if plane[variable] == var:
            
            mm = plane.metal; T1 = plane.T1; z = plane.z; ph = plane.phase
            tt = plane.total_time; cn = plane.config; st = plane.stoich
            
            if ph != 'beta' : continue
            
            data = plane.hops
            
            trap_times = pd.DataFrame(data={'time':hu.trapping_times(data)})
            binned_times,_ = np.histogram(trap_times.time,bins=logbins)
            
            ax.plot(logcenters, binned_times*1e4/tt, label=f'{T1}K')
            ax.legend(title=f'{var} {phases[ph]}')
            
            ## plot guidelines
            if guides :
                if mm == 'Na' and T1 == 600 :
                    ax.plot([15e2, 15e3],[70, 7], lw=0.4, c='k')
                elif mm == 'K' and T1 == 1000 :
                    ax.plot([7e2,4e3],[30,3*7/4], lw=0.4, c='k')
                elif mm == 'Ag' and T1 == 600 :
                    ax.plot([15e2,15e3],[100,10], lw=0.4, c='k')

## make it pretty
for ax in axes:
    ## set scale
    ax.set(xlabel=r'Trapping Lifetime $\tau$, ps', xlim=[0.2,15e4])
    
    ## recolor nicely
    non_guide_lines = [x for x in ax.lines if x.get_label()[0] != '_']
    colors = [batlow_even(j) for j in np.linspace(0, 1, len(non_guide_lines))]            
    for i, l in enumerate(non_guide_lines) : l.set(color=colors[i])
    
    ## remake legend with same title - but possibly new colors
    ax.legend(title=ax.get_legend().get_title().get_text())
    
axes[0].set(xscale='log', yscale='log', ylim=[0.1,None],ylabel='Trapping Events / 10 ns')

fig.tight_layout(pad=0.5, w_pad=0.25)

# =============================================================================
# %% Crowding analysis: load planes. Figures S8-S9
# =============================================================================

# planes_to_plot = all_planes.query('num_planes == 1 & metal == "Na" & config in ["120_4", "120_1"] & T1 == 300 & z == "062"')
planes_to_plot = all_planes.query('num_planes == 1 & metal == "Na" & config in ["120_M4", "120_M1"] & T1 == 300 & z == "062"')

planes_to_plot = planes_to_plot.sort_values(by=['T1','z','metal','stoich'])

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac, do_fill_times)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)    

# =============================================================================
# %% Figures for the crowding piece: maps
# =============================================================================

subject = 'Grid' ## Here: 'Grid'

## assign the variable over which to plot multiple panels. Do pick this.
variable = 'T1'

# ========== automatic things below this line ==========

var2 = 'phase' ## this gets assigned automatically
var_values = set(planes_data[variable]) 

## if a second variable is varied, then find the second variable 
relevant_vars = ['metal', 'config','T1','z','stoich','exclude']
relevant_vars.remove(variable)
if len(planes_to_plot) > len(var_values): 
    for rv in relevant_vars: 
        if len(planes_data[rv].unique()) > 1: var2 = rv; break
    
planes_data.sort_values(by=[var2, variable], inplace=True)
nrows = len(planes_data)//len(var_values)
ncols = len(var_values)

fig, axes = plt.subplots(nrows, ncols,
                         sharex=True, sharey=True, figsize=(ncols*4.05, nrows*3.125))

if len(planes_data) < 2 or len(axes) < 2: axes = [axes]
else : axes = axes.reshape(-1)

## for some reason only works upfront here, not after plotting
#for ax in axes.flat[1:]:
#    axes[0]._shared_y_axes.join(ax,axes[0])
#    axes[0]._shared_x_axes.join(ax,axes[0])

site_pts_list = list()
sites_list = list()
polys_list = list()
mins_list = list()
maxs_list = list()
cell_num_list = list()
boxes_list = list()
voros_list = list()

## iterate through planes to compose lists of quantities to be plotted 
for (i, plane), ax in zip(planes_data.iterrows(), axes):

    data = plane.hops; tt = plane.total_time; ph = plane.phase
    m = plane.metal; T1 = plane.T1; s = plane.stoich; z = plane.z 
    tt = plane.total_time
    BR_sites = plane.BR_sites
    sites_by_r = plane.sites_by_r
    
    ## dist_col is the dataframe column that has the distance index
    dist_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    
    if plane.num_planes > 1: 
        print('This block only works for single planes, not composite planes')
        continue
    
    ## to plot grid maps, I need: site_pts, 2D box, a freud voronoi to continue
    site_pts = np.copy(plane.site_pts)
    site_pts[:,-1] = 0 ## flatten in case it has not done so yet.
    box = freud.box.Box(Lx=plane.cell[0,0], Ly=plane.cell[1,1], is2D=True)
    vor = freud.locality.Voronoi(box)
    
    ## destroy z-coordinate here, and make polygons
    site_pts[:,-1] = 0
    vor.compute((box,site_pts))
    
    ## simple grid geometry: distances from oxygens
    if subject == '# Sites' or subject == 'Grid' :## use distance from oxygens as the coloring value
        sites = data.groupby('new_cell')
        sites = sites[dist_col].mean()
    
    polys_all = [ vor.polytopes[int(i)] for i in list(sites.index) ]
    
    ## save limits and polygons for plotting
    sites_list.append(sites)
    polys_list.append(polys_all)
    mins_list.append(min(sites))
    maxs_list.append(max(sites))
    
    ## also save lists for freud tessellations
    site_pts_list.append(site_pts)
    boxes_list.append(box)
    voros_list.append(vor)
    
## create bounds for coloring below
bounds = [min([x if not np.isnan(x) else 0 for x in mins_list]), max([x if not np.isnan(x) else 0 for x in maxs_list])]
    
## this is the loop that actually creates the figure
for (i,plane), site_vals, polys, ax, site_pts, box, vor \
    in zip(planes_data.iterrows(), sites_list, polys_list, axes, site_pts_list, 
           boxes_list, voros_list) :
    
    plt.sca(ax)
    
    v = str(plane[variable]); v2 = ['', str(plane[var2])][var2 != 'phase']
    m = plane.metal;  T1 = plane.T1;  z = plane.z; ph = plane.phase
    ex = plane.exclude
    
    defect_type = [r'$O_i^{\prime\prime}$', r'$Mg_{Al}^\prime$'][('M' in ex) or (ph == 'bdp')]
    ax_label = f'{defect_type} {ex.replace("M","")}+ sites apart'
    
    Lx = plane.cell[0,0]; Ly = plane.cell[1,1]
    
    ## make up axes labels for the plot
    axt  = ', ' + ['','z='][variable=='z'] +v+ ['','K'][variable=='T1'] 
    axt2 = ', ' + ['','z='][var2=='z']    +v2+ ['','K'][var2 =='T1'] 
    
    ## plot the whole lattice, coloring by property chosen above
    hu.draw_voronoi(box, site_pts, polys, draw_box=True, color_by_property=site_vals, 
                    alpha=0.75, property_clim=bounds,cmap='YlOrRd') 
    ## set xy limits and the such
    ax.set(aspect=1,xlim=(-0.55*Lx,0.55*Lx),ylim=(-0.55*Ly,0.55*Ly),
         xticks=[], yticks=[], ylabel=axt2[2:])

    ## having pre-loaded the defect coordinates separately up top, plot them
    ax.scatter(plane.defects.x, plane.defects.y, c='k', s=30)
    
    ax.set(ylabel=ax_label) ## +axt
    
## make labels: general 2D case
## make top, left, and bottom labels non-empty
for n, ax in enumerate(axes.flat):
    if n >= len(var_values): ## not top row
        ax.set(title='')
    if n % len(var_values) != 0: ## not left column
        ax.set(ylabel='')
    if len(axes.flat) - n > len(var_values) : ## not bottom row
        ax.set(xlabel='')

for i in range(5) : fig.tight_layout(pad=0.5)
    
# =============================================================================
# %% Figure 6: Quenching and crowding in the main text. Load planes.
# =============================================================================

planes_to_plot = all_planes.query('num_planes == 1 & metal == "Na" & config in ["120_4", "120_1"] & T1 == 600 & z == "062"')
# planes_to_plot = all_planes.query('num_planes == 1 & metal == "Na" & config in ["symm_1", "unsym_0"] & T1 == 300 & z == "016"')

planes_to_plot = planes_to_plot.sort_values(by='config',ascending=False)

planes_dicts = []

for plane in planes_to_plot.itertuples(index=False):
        
    ## load a dictionary with all the data and metadata
    this_plane_data = hu.load_plane_with_atoms(plane, frac, do_fill_times)
    
    ## put it into a list
    planes_dicts.append(this_plane_data)
    
planes_data = pd.DataFrame(planes_dicts)    

## I use alpha = 0.75 in maps
from matplotlib import cm
ylorrd = cm.get_cmap('YlOrRd',256)
ylorrd_dim = lambda x : (ylorrd(x)[0], ylorrd(x)[1], ylorrd(x)[2], 0.75)

# =============================================================================
# %% Figure 6: Quenching and crowding in the main text 
# =============================================================================

subject = 'Grid' ## Here: 'Grid', 'Occupancy'
# subject = 'Occupancy'

## assign the variable over which to plot multiple panels. Do pick this.
variable = 'config'

# ========== automatic things below this line ==========

var2 = 'phase' ## this gets assigned automatically
var_values = set(planes_data[variable]) 

## if a second variable is varied, then find the second variable 
relevant_vars = ['metal', 'config','T1','z','stoich','exclude']
relevant_vars.remove(variable)
if len(planes_to_plot) > len(var_values): 
    for rv in relevant_vars: 
        if len(planes_data[rv].unique()) > 1: var2 = rv; break
    
planes_data.sort_values(by=[var2, variable], inplace=True)
nrows = len(planes_data)//len(var_values)
ncols = len(var_values)

fig, axes = plt.subplots(nrows, ncols,
                         sharex=True, sharey=True, figsize=(7.5,3.75))

if len(planes_data) < 2 or len(axes) < 2: axes = [axes]
else : axes = axes.reshape(-1)

## for some reason only works upfront here, not after plotting
#for ax in axes.flat[1:]:
#    axes[0]._shared_y_axes.join(ax,axes[0])
#    axes[0]._shared_x_axes.join(ax,axes[0])

site_pts_list = list()
sites_list = list()
polys_list = list()
mins_list = list()
maxs_list = list()
cell_num_list = list()
boxes_list = list()
voros_list = list()

## iterate through planes to compose lists of quantities to be plotted 
for (i, plane), ax in zip(planes_data.iterrows(), axes):

    data = plane.hops; tt = plane.total_time; ph = plane.phase
    m = plane.metal; T1 = plane.T1; s = plane.stoich; z = plane.z 
    tt = plane.total_time
    BR_sites = plane.BR_sites
    sites_by_r = plane.sites_by_r
    
    ## dist_col is the dataframe column that has the distance index
    dist_col = 'new_ox_r' if ph == 'beta' else 'new_mg_count'
    
    if plane.num_planes > 1: 
        print('This block only works for single planes, not composite planes')
        continue
    
    ## to plot grid maps, I need: site_pts, 2D box, a freud voronoi to continue
    site_pts = np.copy(plane.site_pts)
    site_pts[:,-1] = 0 ## flatten in case it has not done so yet.
    box = freud.box.Box(Lx=plane.cell[0,0], Ly=plane.cell[1,1], is2D=True)
    vor = freud.locality.Voronoi(box)
    
    ## destroy z-coordinate here, and make polygons
    site_pts[:,-1] = 0
    vor.compute((box,site_pts))
    
    ## simple grid geometry: distances from oxygens
    if subject == '# Sites' or subject == 'Grid' :## use distance from oxygens as the coloring value
        sites = data.groupby('new_cell')
        sites = sites[dist_col].mean()
        
    ## occupancies
    if subject == 'Occupancy' :
        toplot_BR = pd.DataFrame(columns=['total',dist_col])
        toplot_aBR = pd.DataFrame(columns=['total',dist_col])
        for r in range(plane.max_r+1): ## range of distances to oxygens / counts of nearby defects
            total_BR, total_aBR = hu.site_occupancies(data,set(sites_by_r[r]),BR_sites,tt,plane,r=r)
            toplot_BR = toplot_BR.append(total_BR,sort=False)
            toplot_aBR = toplot_aBR.append(total_aBR,sort=False)
        sites = toplot_BR.append(toplot_aBR).total
    
    polys_all = [ vor.polytopes[int(i)] for i in list(sites.index) ]
    
    ## save limits and polygons for plotting
    sites_list.append(sites)
    polys_list.append(polys_all)
    mins_list.append(min(sites))
    maxs_list.append(max(sites))
    
    ## also save lists for freud tessellations
    site_pts_list.append(site_pts)
    boxes_list.append(box)
    voros_list.append(vor)
    
## create bounds for coloring below
bounds = [min([x if not np.isnan(x) else 0 for x in mins_list]), max([x if not np.isnan(x) else 0 for x in maxs_list])]
if subject == 'Occupancy' : bounds = [0,1]
    
## this is the loop that actually creates the figure
for (i,plane), site_vals, polys, ax, site_pts, box, vor \
    in zip(planes_data.iterrows(), sites_list, polys_list, axes, site_pts_list, 
           boxes_list, voros_list) :
    
    plt.sca(ax)
    
    v = str(plane[variable]); v2 = ['', str(plane[var2])][var2 != 'phase']
    m = plane.metal;  T1 = plane.T1;  z = plane.z; ph = plane.phase
    ex = plane.exclude
    
    defect_type = [r'$O_i^{\prime\prime}$', r'$Mg_{Al}^\prime$'][('M' in ex) or (ph == 'bdp')]
    ax_label = f'{defect_type} {ex.replace("M","")}+ sites apart'
    
    Lx = plane.cell[0,0]; Ly = plane.cell[1,1]
    
    ## make up axes labels for the plot
    axt  = ', ' + ['','z='][variable=='z'] +v+ ['','K'][variable=='T1'] 
    axt2 = ', ' + ['','z='][var2=='z']    +v2+ ['','K'][var2 =='T1'] 
    
    ## plot the whole lattice, coloring by property chosen above
    hu.draw_voronoi(box, site_pts, polys, draw_box=True, color_by_property=site_vals, 
                    alpha=0.75, property_clim=bounds,cmap='YlOrRd')  ## 
    ## set xy limits and the such
    ax.set(aspect=1,xlim=(-0.55*Lx,0.55*Lx),ylim=(-0.55*Ly,0.55*Ly),
         xticks=[], yticks=[], ylabel=axt2[2:])

    ## having pre-loaded the defect coordinates separately up top, plot them
    if ph != 'beta' or 'M' in cn : 
        spec = {'facecolors':'tab:green', 'edgecolors':'k'}
        half1 = plane.defects.query('-11 < z < -4')
        ax.scatter(half1.x, half1.y, s=20, alpha=1, **spec)
        half2 = plane.defects.query('-11 > z or z > -4')
        ax.scatter(half2.x, half2.y, s=40, alpha=0.5, **spec)
    else : 
        spec = {'facecolors':'tab:red', 'edgecolors':'k', 's':40}
        ax.scatter(plane.defects.x, plane.defects.y, **spec)
    
    # ax.set(ylabel=ax_label) ## +axt
    
## make labels: general 2D case
## make top, left, and bottom labels non-empty
for n, ax in enumerate(axes.flat):
    if n >= len(var_values): ## not top row
        ax.set(title='')
    if n % len(var_values) != 0: ## not left column
        ax.set(ylabel='')
    if len(axes.flat) - n > len(var_values) : ## not bottom row
        ax.set(xlabel='')

for i in range(5) : fig.tight_layout(pad=0.5)
    
    
    
    
    
    
    
    
    