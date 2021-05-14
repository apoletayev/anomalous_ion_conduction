#!/bin/bash
#
#SBATCH --time=40:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --mail-user=apolet@stanford.edu
#SBATCH --mail-type=START,END,FAIL,TIME_OUT

ml python/3.6.1

## USAGE: to take the giant trajectory file and parse it into hops. Magic!
## CHECK: 
## - put this into the folder with the lammps file and the trajectory, and run from there
## - paths to traj_unpack.py and hop_count.py below point to where they actually are
## - the path to the trajectory file matches the first argument to traj_unpack.py
## - the call to hop_count.py points to the right lammps structure file 
##   (same stuff.lmp structure file from which lammps simulated the trajectory)
## EXAMPLE: sbatch -J pNa300 -o ../shell_logs/%j-clever-logfile.out parse-bdp.sh Na unsym 0 300

# echo parameters
echo "metal   = $1"
echo "stoich  = $2"
echo "exclude = $3"
echo "temp    = $4"
echo "starts  = $5"     ## for r2, example: "range(0,9000,100)"
echo "length  = $6"     ## for r2, example: 1000
echo "kwargs  : ${@:7}" ## keyword args for python scripts called from here, e.g. split=10

## possible keywords:
# split= [int] split trajectories when parsing, create atoms w/ shorter runs
# max_time= [ps] only parse trajectories up to this time
# down=50 for downsampling BCF from velocities (only for Burnett scripts)
# first_atom= [int] for parsing mobile-ion trajectories here
# last_atom= [int] for parsing mobile-ion trajectories here

# make up a name for directories where single-atom trajectories & hops will go
DIR=-${1}-${2}_$3-${4}K

date

## Combine the center-of-mass position files from all restarts into one
## This is the "urgent to use" subset of species
## The less-urgent ones are done after everything else.
## For 100ns / 10fs, this is fine with 6G memory. 300ns / 10fs takes 12G.
for species in $1 Al O Mg Oi ; do
    python3 ~/pythonhops/sherlock_combine_restarts.py filenames=cm_$1-$2-$3-${4}K*-$species.fix
done

## compute the center-of-mass velocity autocorrelation function (JACF)
## OK to not pass $5 and $6; those are for a2 only
# sbatch -J j$SLURM_JOB_NAME -o ../shell_logs/%j-jacfs-$1-$2-$3-${4}K.out ~/hop_scripts/parse-parallel.sh $1 $2 $3 $4 $5 $6

## parse the mobile-ion trajectories into single-atom files:
## default parameters (skipped) are first_atom=[1] and last_atom=[12000]
## these may need to change for larger simulations (e.g. THz-pumping ones)

mkdir atoms$DIR

case $2 in
    symm | unsym | oh | unsymLi)
        echo "Stoichiometry $2 looks like it is beta-doubleprime phase."
        python3 ~/pythonhops/traj_unpack_multi.py filepath=./traj_$1-$2-$3-${4}K*-$1.dat folder_out=./atoms${DIR}/ atom=$1 "${@:7}"
        ;;
    "1"*)
        echo "Stoichiometry $2 looks like it is beta phase."
        python3 ~/pythonhops/traj_unpack_multi.py filepath=./traj_$1-$2-$3-${4}K*-$1.dat folder_out=./atoms${DIR}/ atom=$1 "${@:7}"
        ;;
    *)
        echo "unclear stoichiometry for parsing the combined trajetory: $2"
        exit 1
        ;;
esac

## grab the dimensions of the cell
python3 ~/pythonhops/cell_from_traj.py file_in=traj_$1-$2-$3-${4}K-00-$1.dat file_out=cell_$1-$2-$3-${4}K.csv
python3 ~/pythonhops/cell_from_traj.py file_in=traj_$1-$2-$3-${4}K-0-$1.dat file_out=cell_$1-$2-$3-${4}K.csv

date

## adjust starting points for multiple starts if T1 > 1000 (simulation is short)
## THE NUMBERS ARE DIFFERENT FROM NORMAL-LENGTH SIMULATIONS
last_start=50000
duration=50000
pitch=70
if (( $4 > 1051 )) ; then last_start=25000 ; duration=25000; pitch=30 ; fi;

if [ -z $5 ] ; then
    sbatch -J a$1-$2-$3-$4 -o ../shell_logs/%j-parse-a2-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" $duration default
    sbatch -J xy$1-$2-$3-$4 -o ../shell_logs/%j-a2xy-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2-2d.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" $duration default
    sbatch -J vh$1-$2-$3-$4 -o ../shell_logs/%j-vanhove-$1-$2-$3-${4}K.out ~/hop_scripts/calc-vanhove.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" $duration True
    sbatch -J e$1-$2-$3-$4-100 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" 100 eb
    sbatch -J e$1-$2-$3-$4-20 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" 20 eb
    sbatch -J x$1-$2-$3-$4 -o ../shell_logs/%j-dx-$1-$2-$3-${4}K.out ~/hop_scripts/parse-dx.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" $duration default
else 
    sbatch -J a$1-$2-$3-$4 -o ../shell_logs/%j-parse-a2-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2.sh $1 $2 $3 $4 $5 $6 default
    sbatch -J xy$1-$2-$3-$4 -o ../shell_logs/%j-a2xy-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2-2d.sh $1 $2 $3 $4 $5 $6 default
    sbatch -J vh$1-$2-$3-$4 -o ../shell_logs/%j-vanhove-$1-$2-$3-${4}K.out ~/hop_scripts/calc-vanhove.sh $1 $2 $3 $4 $5 $6 True
    sbatch -J e$1-$2-$3-$4-100 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 $5 100 eb
    sbatch -J e$1-$2-$3-$4-20 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 $5 20 eb
    sbatch -J x$1-$2-$3-$4 -o ../shell_logs/%j-dx-$1-$2-$3-${4}K.out ~/hop_scripts/parse-dx.sh $1 $2 $3 $4 $5 $6 default
fi

## correct the mobile-ion CoM to be in the reference frame of the host lattice
## this is ~fast, ~15 minutes on 1 sherlock core (for 10-fs sampling, 100 ns)
case $2 in
    symm | unsym | oh | unsymLi)
        echo "Correcting CoM for an assumed beta-doubleprime phase:"
        python3 ~/pythonhops/correct_cm.py atoms=${1}_bdp_${2}_${3}.lmp template=cm_$1-$2-$3-${4}K
        ;;
    "1"*)
        echo "Correcting CoM for an assumed beta phase with stoichiometry $2 :"
        python3 ~/pythonhops/correct_cm.py atoms=${1}beta${2}_${3}.lmp template=cm_$1-$2-$3-${4}K
        ;;
    *)
        echo "unclear stoichiometry for parsing the combined trajetory: $2"
        exit 1
        ;;
esac

## combine the ensemble a2 files from all restarts into one
## in principle, the msd/nongauss could be subbed for just msd to save space
python3 ~/pythonhops/sherlock_combine_restarts.py filenames=a2_ave_$1-$2-$3-${4}K*-$1.fix

## Count hops : separately for every conduction plane
## Plane names are hard-coded. (a historic artefact of 2019 coding)
## To check whether you have the right coordinates: 
## take fractional z-coordinate of the planes from python,             e.g. z=-0.3333
## add +0.5 to make oordinates like lammps (0 to L, not -L/2 to +L/2), e.g. z=0.1667
## multiply by 100, like percent, round down to integer,               e.g. z=16 (%)
## add a leading zero to have 3 total characters,                      e.g. z=016
## CHECK THAT THE LAMMPS STRUCTURE FILE IS NAMED THE SAME WAY IT IS CALLED BELOW
mkdir hops$DIR

case $2 in 
    symm | unsym | oh | unsymLi)
        for p in 016 049 050 083; do
            python3 ~/pythonhops/hop_count.py $1 $2 $4 $p ./${1}_bdp_${2}_$3.lmp $DIR
        done
        ;;
    "1"*)
        for p in 012 037 062 087; do
            python3 ~/pythonhops/hop_count.py $1 $2 $4 $p ./${1}beta${2}_$3.lmp $DIR
        done
        ;;
    *) 
        echo "unclear stoichiometry for parsing hops: $2"
        exit 1
        ;;
esac

date

## Combine the center-of-mass position files from all restarts into one
## This is the "not urgent to use" subset of species
## The actually urgent ones are done first above
for species in Ob Op plane $1-016 $1-050 $1-083 $1-012 $1-037 $1-062 $1-087 ; do
    python3 ~/pythonhops/sherlock_combine_restarts.py filenames=cm_$1-$2-$3-${4}K*-$species.fix
done

date

# ## compress the original trajectories
# if [ ! -s "traj_${4}K-$1.tar.gz" ] ; then 
#     echo "Compressing trajectories:"
#     tar -cvzf traj_${4}K-$1.tar.gz traj_$1-$2-$3-${4}K*-$1.dat
# else 
#     echo "Looks like the compression target for the trajectory exists."
#     echo "Avoiding an overwrite, skipping compression."
# fi

## parse and remove any RDF files that may have been created
# python3 ~/pythonhops/parse_rdf.py .
# rm *rdf*.fix
# mkdir rdfs
# mv -v *_rdf* rdfs

date
exit 0;