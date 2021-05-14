# anomalous_ion_conduction
Supporting code and sample data for manuscript on anomalous ion transport in beta''- and beta-aluminas.

## Summary

## Contents

## Dependencies

Supporting functions are collected in ```hop_utils.py``` and ```crystal_utils.py```. Module dependencies are:

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [networkx](https://pypi.org/project/networkx/)
- [freud-analysis](https://pypi.org/project/freud-analysis/)
- [deepgraph](https://pypi.org/project/DeepGraph/) (although only deprecated/unpublished analyses)

Current dependencies can be installed via ```requirements.txt``` :
```
pip install -r requirements.txt
```

I also cite [batlow](https://www.fabiocrameri.ch/batlow/) for a perception-optimized color scheme.

## Workflow

### Creating Simulation Structures

Python scripts using ```networkx``` and ```freud``` sample random configurations of defect placements, starting with DL_POLY CONFIG template files. Then LAMMPS input .lmp files are created optimizing for most-representative structures. The two sets of scripts are ```make_repeating_structures_[phase].py``` and ```make_structure_[phase].py```, where phase is beta, beta'', and Mg-doped beta. The outputs are under ```structures```.

### LAMMPS Simulation

Sample LAMMPS input files are included. These are easy to modify for multiple mobile ions and temperatures using only variable declarations at the beginnings of the files. Because of (sherlock)[https://sherlock.stanford.edu] job duration limitations, (restarting)[https://lammps.sandia.gov/doc/restart.html] and recursion was used for long simulations. Shell scripts for slurm are included. 

### Processing and Statistics

### Figures
