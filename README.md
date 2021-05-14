# Simulation and Analyses for "Defect-Driven Anomalous Transport in Fast-Ion Conducting Solid Electrolytes"

## Summary

This reponsitory contains the code scripts and sample data for the manuscript "Defect-Driven Anomalous Transport in Fast-Ion Conducting Solid Electrolytes", which is undergoing peer review as of May 13, 2021.

The large-scale classical molecular dynamics simulations are run via [LAMMPS](https://lammps.sandia.gov). Full datasets are not included.

Disclaimer: we do not claim this is good code. This work has been a learning experience more than anything else.

## Contents

- ```sample_data```: two sample datasets, one for Na beta"-alumina, and one for Ag beta-alumina.
- ```sherlock_scripts```: shell, python, and LAMMPS scripts for running simulations and extracting hopping events and statistical descriptors of transport.
- ```structures```: sample crystal structures with quasi-randomly generated distributions of defect atoms.
- ```templates```: template files for generating crystal structures.
- ```utils```: supporting python functions.
- python scripts: generating crystal structures for simulation, and making figures in the manuscript.

## Dependencies and Used Codes

Python module dependencies are:

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [networkx](https://pypi.org/project/networkx/)
- [freud-analysis](https://pypi.org/project/freud-analysis/)
- [seaborn](https://pypi.org/project/seaborn/)

Python dependencies can be installed via ```requirements.txt``` :
```
pip install -r requirements.txt
```

Parallelization of several analyses is done with [parallel](https://www.gnu.org/software/parallel/).

We also cite [batlow](https://www.fabiocrameri.ch/batlow/) for a perception-optimized color scheme.

## Workflow

Python scripts are organized similar to jupyter notebooks, and are best run in blocks/cells using something like [spyder](https://www.spyder-ide.org/).

### Creating Simulation Structures

Python scripts using ```networkx``` and ```freud``` sample random configurations of defect placements, starting with DL_POLY CONFIG template files. Then LAMMPS input .lmp files are created optimizing for most-representative structures. The two sets of scripts are ```make_repeating_structures_[phase].py``` and ```make_structure_[phase].py```, where phase is beta, beta'', and Mg-doped beta. The outputs are under ```structures```.

### LAMMPS Simulation

Sample LAMMPS input files are included. These are easy to modify for multiple mobile ions and temperatures using only variable declarations at the beginnings of the files. Because of [sherlock](https://sherlock.stanford.edu) job duration limitations, [restarting](https://lammps.sandia.gov/doc/restart.html) was used for 100-nanosecond-long simulations. Shell scripts for slurm are included.

### Processing Trajectories

LAMMPS trajectories for mobile ions and their center of mass are parsed to individual atom trajectories. Then, hopping events within each two-dimensional conduction plane are identified as boundary crossings using the input LAMMPS crystal structures; this relies on the two-dimensional nature of conduction and ignores small motions of the host atoms. Then the hopping events from individual conduction planes within each simulation are combined.

Statistical descriptors of transport are computed from the trajectories of mobile ions: mean-square displacements and [non-Gaussian parameter](sherlock_scripts/shell_scripts/parse-r2a2-2d.sh) in two dimensions, [distribution of displacements](sherlock_scripts/shell_scripts/parse-dx.sh) along [100], self part of the [van Hove](sherlock_scripts/shell_scripts/calc-vanhove.sh) function, and [ergodicity breaking](sherlock_scripts/shell_scripts/parse-eb.sh) parameter.

Python scripts and shell scripts for this are included in ```sherlock_scripts```.

### Manuscript Figures

Python code for manuscript figures is included ```make_structure_[micro/macro].py```. The "micro" script deals with atomistic hopping events, and the "macro" script deals with statistical descriptors.

The "micro" script compiles a .csv listing of all simulation planes, which is used by the "macro" script.

As it is not possible to include all data via GitHub, most figure-making blocks/cells will not immediately run.

## Sample Data

Two sets of sample data are included. These are limited by GitHub constraints on the size of data files.
- Na beta''-alumina 300 K. 20 ns of center-of-mass trajectory, 20 ns of hopping events, extracted statistical descriptors.
- Ag beta-alumina 300 K. 20 ns of center-of-mass trajectory, 10 ns of hopping events, extracted statistical descriptors.
