#!/bin/bash
#
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=16
#SBATCH --mem=96G
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
## EXAMPLE: sbatch -J rN2-0-300 -o ../shell_logs/%j-r2-stuff.out parse-r2a2.sh Na 102 0 300 "range(0,9000,100)" 1000 default

## HARDCODED HERE:
## - the fact that there are 4 * stoich atoms total


# echo parameters
echo "metal   = $1"
echo "stoich  = $2"
echo "exclude = $3"
echo "temp    = $4"
echo "starts  = $5" ## for r2, example: "range(0,9000,100)"
echo "length  = $6" ## for r2, example: 1000
echo "t bins  = $7"

# make up a name for directories where single-atom trajectories, velocities, & hops will go
DIR=-${1}-${2}_$3-${4}K
mkdir outputs
mkdir logs

## security against sherlock lags: replicate this job with an "afternotok" dependency
sbatch -J $SLURM_JOB_NAME-redo -o ../shell_logs/%j-$SLURM_JOB_NAME-redo.out --dependency=afternotok:$SLURM_JOB_ID ~/hop_scripts/parse-r2a2-2d.sh $1 $2 $3 $4 $5 $6 $7


##################
# r2, r4, and a2 #
##################

## to check for completion, count files: files=` ls r2s$DIR | wc -l`

mkdir r2d$DIR
rd=$6 ## duration for r2, r4, a2

if [ "$7" = "short" ] ; then ATOMS="short_atoms" ; else ATOMS="atoms" ; fi

atom_files=`ls ${ATOMS}${DIR} | wc -l`
r2_files=`ls r2d$DIR | grep "${rd}ps\.csv" | wc -l`
num_tries=1

echo "Found $r2_files files"

## define each srun to only use one core
srun="srun --exclusive -N1 -n1 --mem=6g"

## define parallel to not start more than ntasks processes at a time
parallel="parallel --delay 0.2 --jobs $SLURM_NTASKS --joblog logs/r2d-$1-$2-$3-$4-${rd}ps.log --resume-failed --timeout 300%"

## check completion in case the thing times out
while (( $r2_files < $atom_files ))
do
    echo "Starting r2s, try $num_tries, $(date)"
    
    ## calculate all r2's and r4's using multiple starts
    $parallel "$srun python3 ~/pythonhops/sherlock_multistart.py file={1} \
    starts='$5' cell_file=cell_$1-$2-$3-${4}K.csv duration=$rd \
    file_out=./r2d$DIR/{1/.}-r2d-${rd}ps.csv timebins=$7 2d=True " ::: ${ATOMS}${DIR}/*.dat
    
    ## count files again
    r2_files=`ls r2d$DIR | grep "${rd}ps\.csv" | wc -l`
    echo "After try $num_tries, there are $r2_files files"
    
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

echo 'done with the parallel part, starting to combine:' 
date

## combine r2s, r4s, a2s
python3 ~/pythonhops/sherlock_combine.py folder=./r2d$DIR template=/*r2d*${rd}ps.csv \
file_out=./outputs/${1}-${2}-${3}-a2xy-${4}K-${rd}ps.csv 

# ## calculate the self part of a van Hove function
# python3 ~/pythonhops/sherlock_combine.py option=vanhove folder=./r2s$DIR \
# file_out=./outputs/${1}-${2}-${3}-gs-${4}K-${rd}ps.csv template=/*r2*${rd}ps.csv

exit 0; 