#!/bin/bash
#
#SBATCH --time=01:20:00
#SBATCH --partition=normal
#SBATCH --ntasks=16
#SBATCH --mem=128g
#SBATCH --mail-user=apolet@stanford.edu
#SBATCH --mail-type=START,END,FAIL,TIME_OUT

ml python/3.6.1 py-schwimmbad
ml system parallel/20180122

## USAGE: parallelized MD workup calculations:
## - single-atom trajectories and average r2 and non-Gaussian a2.
## - JACF
## - VACF
## CHECK: 
## - put this into the folder with the atoms folder, and run from there
## - path to sherlock_multistart.py below points to where it actually is
## - the DIR variable points to the atoms folder
## - 5th input is in doublequotes and is interpretable python, e.g. "range(0,9000,100)"
## - 6th (last) input is the duration of every trajectory
## EXAMPLE: sbatch -J r-Na-300 -o ../shell_logs/%j-r2-parallel-stuff.out parse-r2-parallel.sh Na 120 4 300 "range(0,9000,100)" 1000

## HARDCODED HERE:
## - the fact that there are 4 * stoich atoms total


# echo parameters
echo "metal   = $1"
echo "stoich  = $2"
echo "exclude = $3"
echo "temp    = $4"
echo "starts  = $5" ## for r2, example: "range(0,9000,100)"
echo "length  = $6" ## for r2, example: 1000
echo "long    = $7" ## use True for long (100ns) simulations

# make up a name for directories where single-atom trajectories, velocities, & hops will go
DIR=-${1}-${2}_$3-${4}K
mkdir outputs
mkdir logs

## security against sherlock lags: replicate this job with an "afternotok" dependency
sbatch -J $SLURM_JOB_NAME-redo -o ../shell_logs/%j-$SLURM_JOB_NAME-repeat-of-$SLURM_JOB_ID.out --dependency=afternotok:$SLURM_JOB_ID ~/hop_scripts/calc-vanhove.sh $1 $2 $3 $4 $5 $6 $7

##################
# r2, r4, and a2 #
##################

## to check for completion, count files: files=` ls r2s$DIR | wc -l`

mkdir vhs$DIR
rd=$6 ## duration for van Hove
atom_files=`ls atoms$DIR | wc -l`
vh_files=`ls vhs$DIR | grep "${rd}ps\.csv" | wc -l`
num_tries=1

echo "Found $vh_files files"

## define each srun to only use one core
srun="srun --exclusive -N1 -n1 --mem=6g"

## define parallel to not start more than ntasks processes at a time
parallel="parallel --delay 0.2 --jobs $SLURM_NTASKS --joblog logs/vh-$1-$2-$3-$4-${rd}ps.log --resume-failed --timeout 300%"

## check completion in case the thing times out
while (( $vh_files < $atom_files ))
do
    echo "Starting vhs, try $num_tries, $(date)"
    
    ## calculate binned r2's using multiple starts
    $parallel "$srun python3 ~/pythonhops/sherlock_multistart.py file={1} \
    starts='$5' duration=$rd cell_file=cell_$1-$2-$3-${4}K.csv \
    file_out=./vhs$DIR/{1/.}-vh-${rd}ps.csv option=vanhove long=$7" ::: atoms$DIR/*.dat
    
    ## count files again
    vh_files=`ls vhs$DIR | grep "${rd}ps\.csv" | wc -l`
    echo "After try $num_tries, there are $vh_files files"
    
    ## augment number of tries and break glass in case of emergency
    num_tries=$((num_tries+1))
    if (( $num_tries > 5 )) ; then exit 1; fi;
done

## cancel the backup job
scancel --name $SLURM_JOB_NAME-redo

########
# VACF #
########

########
# JACF #
########

########################
# combine and van Hove #
########################

## security against sherlock lags: replicate this job with an "afternotok" dependency
sbatch --dependency=afternotok:$SLURM_JOB_ID -J $SLURM_JOB_NAME-combine -o ../shell_logs/%j-combine-$SLURM_JOB_NAME-$SLURM_JOB_ID.out ~/hop_scripts/combine-bigmem.sh $1 $2 $3 $4 $5 $6 vanhove

echo 'done with the parallel part, starting to combine:' 
date

# ## combine r2s, r4s, a2s
# python3 ~/pythonhops/sherlock_combine.py folder=./r2s$DIR template=/*r2*${rd}ps.csv \
# file_out=./outputs/${1}-${2}-${3}-a2-${4}K-${rd}ps.csv 

## calculate the self part of a van Hove function
python3 ~/pythonhops/sherlock_combine.py option=vanhove folder=./vhs$DIR \
file_out=./outputs/${1}-${2}-${3}-gs-${4}K-${rd}ps.csv template=/*vh*${rd}ps.csv

exit 0; 