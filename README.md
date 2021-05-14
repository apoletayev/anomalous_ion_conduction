# anomalous_ion_conduction

## Summary

Supporting code and sample data for manuscript on anomalous ion transport in beta''- and beta-aluminas.

## Dependencies

Supporting functions are collected in ```hop_utils.py``` and ```crystal_utils.py```. Module dependencies are:

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [networkx](https://pypi.org/project/networkx/)
- [freud-analysis](https://pypi.org/project/freud-analysis/)
- [seaborn](https://pypi.org/project/seaborn/)

Dependencies can be installed via ```requirements.txt``` :
```
pip install -r requirements.txt
```

I also cite [batlow](https://www.fabiocrameri.ch/batlow/) for a perception-optimized color scheme.

## Workflow

Python scripts are organized similar to jupyter notebooks, and are best run in blocks/cells using something like [spyder](https://www.spyder-ide.org/).

### Creating Simulation Structures

Python scripts using ```networkx``` and ```freud``` sample random configurations of defect placements, starting with DL_POLY CONFIG template files. Then LAMMPS input .lmp files are created optimizing for most-representative structures. The two sets of scripts are ```make_repeating_structures_[phase].py``` and ```make_structure_[phase].py```, where phase is beta, beta'', and Mg-doped beta. The outputs are under ```structures```.

### LAMMPS Simulation

Sample LAMMPS input files are included. These are easy to modify for multiple mobile ions and temperatures using only variable declarations at the beginnings of the files. Because of [sherlock](https://sherlock.stanford.edu) job duration limitations, [restarting](https://lammps.sandia.gov/doc/restart.html) and recursion was used for long simulations. Shell scripts for slurm are included.

### Processing Trajectories

LAMMPS trajectories for mobile ions and their center of mass are parsed to individual atom trajectories. Then, hopping events are identified as boundary crossings using the input LAMMPS crystal structures; this relies on the two-dimensional nature of conduction and neglects small motions of the host atoms.

Statistical descriptors of transport are computed from the trajectories of mobile ions: mean-square displacements, non-Gaussian parameter, distribution of displacements along [100], self part of the van Hove function, ergodicity breaking parameter.

Python scripts and shell scripts for this are included.

### Manuscript Figures

Python code for manuscript figures is included ```make_structure_[micro/macro].py```. The "micro" script deals with atomistic hopping events, and the "macro" script deals with statistical descriptors.

The "micro" script compiles a .csv listing of all simulation planes, which is used by the "macro" script.

As it is not possible to include all data, most figure-making blocks/cells will not immediately run.

## Sample Data

Two sets of sample data are included. These are limited by GitHub constraints on the size of data files.
- Na beta''-alumina 300 K. 20 ns of center-of-mass trajectory, 20 ns of hopping events, extracted statistical descriptors.
- Ag beta-alumina 300 K. 20 ns of center-of-mass trajectory, 10 ns of hopping events, extracted statistical descriptors.
