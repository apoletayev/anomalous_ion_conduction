#!/bin/bash
#
#SBATCH --time=47:59:59
#SBATCH --ntasks=128
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=apolet@stanford.edu
#SBATCH --mail-type=START,END,FAIL,TIME_OUT

## USAGE: sbatch -J jobname -o %j-output-file-name.out steady-recursion128.sh Na 120 4 300 "range(0,50000,70)" 50000 0
## here, everything following the .sh filename are input parameters, $1 becomes Na.

## this loads the lammps module
ml chemistry lammps/20200303

echo "metal     = $1"
echo "rule      = $2"
echo "exclude   = $3"
echo "temp      = $4"
echo "starts    = $5" ## for long r2-a2, example: "range(0,50000,70)". Numpy's arange() is recognized.
echo "length    = $6" ## for long r2-a2, example: 50000
echo "iteration = $7" ## for recursive calling of lammps; 0 to start, 1+ to continue from a 
echo "kwargs  : ${@:8}" ## keyword args for python scripts called from here, esp. timestep if not 1 fs

## possible keywords:
# split= [int] split trajectories when parsing, create atoms w/ shorter runs
# max_time= [ps] only parse trajectories up to this time (both hops & a2)
# down=50 for downsampling BCF from velocities (only for Burnett scripts)
# first_atom= [int] for parsing mobile-ion trajectories
# last_atom= [int] for parsing mobile-ion trajectories

## augment iteration number, and make one with a leading zero
iter=$(($7+1))
printf -v iter_lmp "%02g" $7

if (($7 < 1)) ; then 
    ## start a recursive job with dependency on this one
    sbatch -J ${SLURM_JOB_NAME::-2}-$iter --dependency=afternotok:$SLURM_JOB_ID -o ../shell_logs/%j-run-recursion-$1-$2-$3-${4}K-$iter.out ~/hop_scripts/steady-recursion128.sh $1 $2 $3 $4 $5 $6 $iter "${@:8}"
    
    ## start a job to parse the short-time trajectory
    sbatch -J p$SLURM_JOB_NAME-short --dependency=afternotok:$SLURM_JOB_ID -o ../shell_logs/%j-parse-short-$1-$2-$3-${4}K.out ~/hop_scripts/parse-short.sh $1 $2 $3 $4
    
    ## start parsing jobs in case this is a "normal-length" simulation 
    ## i.e. there will not be a need to run another one recursively
    sbatch -J p$SLURM_JOB_NAME --dependency=afterok:$SLURM_JOB_ID -o ../shell_logs/%j-parse-hops-$1-$2-$3-${4}K.out ~/hop_scripts/parse-multi.sh $1 $2 $3 $4 $5 $6 "${@:8}"
    
    ## run lammps with starter file
    srun lmp_mpi -in in.analysis-$1-$2-$3-${4}K-starter -log log.$1-$2-$3-${4}K-$7 -var iter ${iter_lmp}
    
elif (($7 < 10)) ; then
    ## start a job with not-ok (timeout only) dependency on this one. 
    ## This is to restart LAMMPS if the simulation time is not reached
    sbatch -J ${SLURM_JOB_NAME::-2}-$iter --dependency=afternotok:$SLURM_JOB_ID -o ../shell_logs/%j-run-recursion-$1-$2-$3-${4}K-$iter.out ~/hop_scripts/steady-recursion128.sh $1 $2 $3 $4 $5 $6 $iter "${@:8}"
    
    ## parse trajectories into hops. This can be run separately, takes ~16 hours for 100ns 
    ## parsing first combines all relevant files generated from all iterations
    sbatch -J p${SLURM_JOB_NAME::-2} --dependency=afterok:$SLURM_JOB_ID -o ../shell_logs/%j-parse-hops-$1-$2-$3-${4}K.out ~/hop_scripts/parse-multi.sh $1 $2 $3 $4 $5 $6 "${@:8}"
    
    ## run lammps with continuation file
    srun lmp_mpi -in in.analysis-$1-$2-$3-${4}K-ongoing -log log.$1-$2-$3-${4}K-$7 -var iter ${iter_lmp}
    
    ## sound off on being done - this will only run if lammps does not time out
    echo "$(date) : iteration $7 finished w/ LAMMPS, starting analysis scripts"

    ## cancel the next iteration
    scancel --name ${SLURM_JOB_NAME::-2}-$iter
elif (($7 < 20)) ; then
    ## start a job with not-ok (timeout only) dependency on this one. 
    ## This is to restart LAMMPS if the simulation time is not reached
    sbatch -J ${SLURM_JOB_NAME::-3}-$iter --dependency=afternotok:$SLURM_JOB_ID -o ../shell_logs/%j-run-recursion-$1-$2-$3-${4}K-$iter.out ~/hop_scripts/steady-recursion128.sh $1 $2 $3 $4 $5 $6 $iter "${@:8}"
    
    ## parse trajectories into hops. This can be run separately, takes ~16 hours for 100ns 
    ## parsing first combines all relevant files generated from all iterations
    sbatch -J p${SLURM_JOB_NAME::-3} --dependency=afterok:$SLURM_JOB_ID -o ../shell_logs/%j-parse-hops-$1-$2-$3-${4}K.out ~/hop_scripts/parse-multi.sh $1 $2 $3 $4 $5 $6 "${@:8}"
    
    ## run lammps with continuation file
    srun lmp_mpi -in in.analysis-$1-$2-$3-${4}K-ongoing -log log.$1-$2-$3-${4}K-$7 -var iter ${iter_lmp}
    
    ## sound off on being done - this will only run if lammps does not time out
    echo "$(date) : iteration $7 finished w/ LAMMPS, starting analysis scripts"

    ## cancel the next iteration
    scancel --name ${SLURM_JOB_NAME::-3}-$iter
else 
    ## run analysis scripts
    echo "got to the $7 iteration (should only happen if the simulation ran out of time in the last iteration)."
    echo "But starting analysis just in case."
    
    ## parse trajectories into hops. This can be run separately, takes ~1 hour 
    sbatch -J p$SLURM_JOB_NAME -o ../shell_logs/%j-parse-hops-$1-$2-$3-${4}K.out ~/hop_scripts/parse-multi.sh $1 $2 $3 $4 $5 $6 "${@:8}"
    
fi

exit 0