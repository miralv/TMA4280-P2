#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --mem=23000 
#SBATCH --ntasks-per-node=2	### Number of tasks (MPI Processes)
#SBATCH --cpus-per-task=1	### Number of threads per task (OMP threads)
#SBATCH --job-name="speedup_test"
#SBATCH --output="b2_speedup.out"
#SBATCH --mail-user=miralv@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} 

### Test the speedup
module purge
module load GCC/5.4.0-2.26
module load CMake/3.5.2
module load OpenMPI/1.10.3

cmake ..
make

###skriver ut n error duration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
	
n=$(echo $((2**14)))
mpirun -np 2 ./poisson_parallel $n $OMP_NUM_THREADS
