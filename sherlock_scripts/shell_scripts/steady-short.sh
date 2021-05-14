#!/bin/bash
#
#SBATCH --time=00:25:00
#SBATCH --ntasks=64
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=apolet@stanford.edu
#SBATCH --mail-type=START,END,FAIL,TIME_OUT

## USAGE: sbatch -J jobname -o %j-output-file-name.out steady64.sh Na 120 4 300
## here, everything following the .sh filename are input parameters, $1 becomes Na.

## this loads the lammps module
ml chemistry lammps/20200303

echo 'metal   =' $1
echo 'rule    =' $2
echo 'exclude =' $3
echo 'temp    =' $4
echo 'starts  =' $5 ## OPTIONAL for a2, example: "arange(0,9000,10)"
echo 'length  =' $6 ## OPTIONAL for a2, example: 1000

date
echo "Starting LAMMPS, $(date)"

## NOTE: CHECK that your in.file is named this way: metal-rule-temperature + Kelvin
## Within each in.file, CHECK all the input parameter variables (metal, T1, rule, exclude)
## Within each in.file, also check the total length of simulation in the 'run' commands
srun lmp_mpi -in in.analysis-$1-$2-$3-${4}K-short -log log.$1-$2-$3-${4}K

echo "Finished LAMMPS, starting new jobs to parse everything $(date)"

## parse trajectories into hops. This can be run separately, takes ~1 hour 
sbatch -J p$1-$2-$3-$4 -o ../shell_logs/%j-parse-hops-$1-$2-$3-${4}K.out ~/hop_scripts/parse-short.sh $1 $2 $3 $4 $5 $6

## compute center-of-mass velocity autocorrelations 
## OK to not pass $5 and $6; those are for a2
# sbatch -J j$1-$2-$3-$4 -o ../shell_logs/%j-jacfs-$1-$2-$3-${4}K.out ~/hop_scripts/parse-parallel.sh $1 $2 $3 $4 $5 $6

## parse velocities for multiple species
sbatch -J v$1-$2-$3-$4 -o ../shell_logs/%j-parse-vels-$1-$2-$3-${4}K.out ~/hop_scripts/parse-vels-short.sh $1 $2 $3 $4

exit 0