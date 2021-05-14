#!/bin/bash
#
#SBATCH --time=01:15:00
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
## EXAMPLE: sbatch -J eN2-0-300 -o ../shell_logs/%j-eb-stuff.out parse-eb.sh Na 102 0 300 "range(0,9000,100)" 1000

## HARDCODED HERE:
## - the fact that there are 4 * stoich atoms total


# echo parameters
echo "metal   = $1"
echo "stoich  = $2"
echo "exclude = $3"
echo "temp    = $4"
echo "starts  = $5" ## for EB, example: "range(0,9000,100)". Overridden by $7
echo "length  = $6" ## for EB, example: 1000
echo "t bins  = $7" ## overrides $5, shorthand for log bins: $7=eb

# make up a name for directories where single-atom trajectories, velocities, & hops will go
DIR=-${1}-${2}_$3-${4}K
mkdir outputs
mkdir logs

## security against sherlock lags: replicate this job with an "afternotok" dependency
sbatch -J $SLURM_JOB_NAME-redo -o ../shell_logs/%j-$SLURM_JOB_NAME-redo-of-$SLURM_JOB_ID.out --dependency=afternotok:$SLURM_JOB_ID ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 $5 $6 $7

################
# EB parameter #
################

## to check for completion, count files: files=` ls ebs$DIR | wc -l`

mkdir ebs$DIR
delta=$6 ## duration for ergodicity breaking here
atom_files=`ls atoms$DIR | wc -l`
eb_files=`ls ebs$DIR | grep "${6}ps\.csv" | wc -l`
num_tries=1

echo "Found $eb_files files"

## define each srun to only use one core and reserve memory
srun="srun --exclusive -N1 -n1 --mem=6g"

## define parallel to not start more than ntasks processes at a time
parallel="parallel --delay 0.2 --jobs $SLURM_NTASKS --joblog logs/eb-$1-$2-$3-$4-${delta}ps.log --resume-failed --timeout 300%"

## check completion in case the thing times out
while (( $eb_files < $atom_files ))
do
    echo "Starting EBs, try $num_tries, $(date)"
    
    ## calculate all single-atom diffusion coeffs using multiple starts
    $parallel "$srun python3 ~/pythonhops/sherlock_multistart.py option=eb \
    timebins=$7 cell_file=cell_$1-$2-$3-${4}K.csv starts='$5' \
    file={1} duration=$delta file_out=./ebs$DIR/{1/.}-eb-${delta}ps.csv" ::: atoms$DIR/*.dat
    
    ## count files again
    eb_files=`ls ebs$DIR | grep "${6}ps\.csv" | wc -l`
    echo "After try $num_tries, there are $eb_files files"
    
    ## augment number of tries and break glass in case of emergency
    num_tries=$((num_tries+1))
    if (( $num_tries > 5 )) ; then exit 1; fi;
done

## backup job is no longer necessary
scancel --name $SLURM_JOB_NAME-redo

########
# VACF #
########

########
# JACF #
########

###########
# combine #
###########

echo 'done with the parallel part, starting to combine:' 
date

## combine single-atom diffusion coefficients into the EB parameter
python3 ~/pythonhops/sherlock_combine.py duration=$delta option=eb \
folder=./ebs$DIR file_out=./outputs/${1}-${2}-${3}-eb-${4}K-${delta}ps.csv template=/*eb*${delta}ps.csv

exit 0; 