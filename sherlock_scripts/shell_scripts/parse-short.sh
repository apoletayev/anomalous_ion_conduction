#!/bin/bash
#
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
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
echo 'metal   =' $1
echo 'stoich  =' $2
echo 'exclude =' $3
echo 'temp    =' $4
echo 'starts  =' $5 ## OPTIONAL, example: "range(0,5000,7)"
echo 'length  =' $6 ## OPTIONAL, example: 5000
echo "kwargs  : ${@:7}" ## keyword args for python scripts called from here, e.g. max_time 


# make up a name for directories where single-atom trajectories & hops will go
DIR=-${1}-${2}_$3-${4}K

date

## check for the trajectory existing and being named the right way.
if [ ! -s "traj_$1-$2-$3-${4}K-$1-short.dat" ] ; then
    if [ -s "traj_$1-$2-$3-${4}K-0-$1-short.dat" ] ; then 
        mv traj_$1-$2-$3-${4}K-0-$1-short.dat traj_$1-$2-$3-${4}K-$1-short.dat
        echo "Found a traj from recursion, traj_$1-$2-$3-${4}K-0-$1-short.dat , and renamed it"
    else 
        echo "Did not find a trajectory file."
    fi
else 
    echo "Found a trajectory : traj_$1-$2-$3-${4}K-$1-short.dat "
fi

## parse the one big trajectory into single-atom files:
## this is done in several steps so as not to overload the linux limit on open files (512)
## the averaging (third call) is not strictly necessary, but good to have just in case.
## The number 600 is the index (in the lammps structure file) of the last mobile ion.
## As that number is currently hard-coded, it is prone to errors if the structure file changes.
## SO CHECK THAT THE INDICES OF MOBILE IONS MATCH THESE INDICES.
## The number 12000 is a catch-all for all atoms; it could well be 600 as only the 
## mobile ions are counted - but 12000 will need to be changed fewer times. 
mkdir short_atoms$DIR
case $2 in
    unsym | symm | oh | unsymLi)
        echo "Stoichiometry $2 looks like it is beta-doubleprime phase."
        # python3 ~/pythonhops/traj_unpack.py ./traj_$1-$2-$3-${4}K-$1-short.dat ./short_atoms${DIR}/ 1 12000 $1
        python3 ~/pythonhops/traj_unpack_multi.py filepath=./traj_$1-$2-$3-${4}K-$1-short.dat folder_out=./short_atoms${DIR}/ atom=$1 "${@:7}"
        ;;
    "1"*)
        echo "Stoichiometry $2 looks like it is beta phase."
        # python3 ~/pythonhops/traj_unpack.py ./traj_$1-$2-$3-${4}K-$1-short.dat ./short_atoms${DIR}/ 1 12000 $1
        python3 ~/pythonhops/traj_unpack_multi.py filepath=./traj_$1-$2-$3-${4}K-$1-short.dat folder_out=./short_atoms${DIR}/ atom=$1 "${@:7}"
        ;;
    *)
        echo "Stoichiometry $2 is not clear, cannot parse trajectory into atoms"
        exit 1
        ;;
esac

## grab the dimensions of the cell
python3 ~/pythonhops/cell_from_traj.py file_in=traj_$1-$2-$3-${4}K-$1-short.dat file_out=cell_$1-$2-$3-${4}K.csv

date

## starting points for multiple starts (simulation is short)
last_start=90
duration=10
pitch=0.05
# if (( $4 > 1051 )) ; then last_start=2500 ; duration=2500; pitch=4 ; fi;

if [ -z $5 ] ; then
    sbatch -J a$1-$2-$3-$4 -o ../shell_logs/%j-parse-a2-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2.sh $1 $2 $3 $4 "arange(0,$last_start,$pitch)" $duration short
    sbatch -J xy$1-$2-$3-$4 -o ../shell_logs/%j-a2xy-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2-2d.sh $1 $2 $3 $4 "arange(0,$last_start,$pitch)" $duration short
    # sbatch -J vh$1-$2-$3-$4 -o ../shell_logs/%j-vanhove-$1-$2-$3-${4}K.out ~/hop_scripts/calc-vanhove.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" $duration False
    # sbatch -J e$1-$2-$3-$4 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" 20 eb
    # sbatch -J e$1-$2-$3-$4 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 "range(0,$last_start,$pitch)" 100 eb
    # parsing dx would also be here if this were not a "short" script. 
else 
    sbatch -J a$1-$2-$3-$4 -o ../shell_logs/%j-parse-a2-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2.sh $1 $2 $3 $4 $5 $6 short
    sbatch -J xy$1-$2-$3-$4 -o ../shell_logs/%j-a2xy-$1-$2-$3-${4}K.out ~/hop_scripts/parse-r2a2-2d.sh $1 $2 $3 $4 $5 $6 short
    # sbatch -J vh$1-$2-$3-$4 -o ../shell_logs/%j-vanhove-$1-$2-$3-${4}K.out ~/hop_scripts/calc-vanhove.sh $1 $2 $3 $4 $5 $6 False
    # sbatch -J e$1-$2-$3-$4 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 $5 20 eb
    # sbatch -J e$1-$2-$3-$4 -o ../shell_logs/%j-eb-$1-$2-$3-${4}K.out ~/hop_scripts/parse-eb.sh $1 $2 $3 $4 $5 100 eb
    # parsing dx would also be here if this were not a "short" script. 
fi

## compress the original trajectory
if [ ! -s "traj_${4}K-$1-short.tar.gz" ] ; then 
    echo "Compressing short trajectory:"
    tar -cvzf traj_${4}K-$1-short.tar.gz traj_$1-$2-$3-${4}K-$1-short.dat
    # rm traj_$1-$2-$3-${4}K-$1-short.dat
else 
    echo "Looks like the compression target for the trajectory exists."
    echo "Avoiding an overwrite, skipping compression."
fi
# tar -cvzf vels_${4}K-$1.tar.gz vels_$1-$2-$3-${4}K-$1.dat

## Count hops : separately for every conduction plane
## Plane names are hard-coded. (a historic artefact of 2019 coding)
## To check whether you have the right coordinates: 
## take fractional z-coordinate of the planes from python,             e.g. z=-0.3333
## add +0.5 to make oordinates like lammps (0 to L, not -L/2 to +L/2), e.g. z=0.1667
## multiply by 100, like percent, round down to integer,               e.g. z=16 (%)
## add a leading zero to have 3 total characters,                      e.g. z=016
## CHECK THAT THE LAMMPS STRUCTURE FILE IS NAMED THE SAME WAY IT IS CALLED BELOW
# mkdir hops$DIR
# for p in 012 037 062 087; do
#     python3 ~/pythonhops/hop_count.py $1 $2 $4 $p ./${1}beta${2}_$3.lmp $DIR
# done

# date

# ## parse and remove any RDF files that may have been created
# python3 ~/pythonhops/parse_rdf.py .
# rm *rdf*.fix
# mkdir rdfs
# mv -v *_rdf* rdfs

date
exit 0;