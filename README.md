# anomalous_ion_conduction
Supporting code and sample data for manuscript on anomalous ion transport in beta''- and beta-aluminas.

## Summary

## Contents

## Dependencies

- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [networkx](https://pypi.org/project/networkx/)
- [freud-analysis](https://pypi.org/project/freud-analysis/)

Dependencies can be installed via ```requirements.txt``` : ```pip install -r requirements.txt```

## Workflow

### Creating Simulation Structures

Python scripts using ```networkx``` and ```freud``` sample random configurations of defect placements, starting with DL_POLY CONFIG template files. Then LAMMPS input .lmp files are created optimizing for most-representative structures. The two sets of scripts are ```make_repeating_structures_[phase].py``` and ```make_structure_[phase].py```. The outputs are under ```structures/```.

### LAMMPS Simulation

### Processing and Statistics

### Figures
