#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --mem=15000
#SBATCH --ntasks-per-node=6	### Number of tasks (MPI Processes)
#SBATCH --cpus-per-task=6	### Number of threads per task (OMP threads)
#SBATCH --job-name="Pt_6"
#SBATCH --output="P_6.out"
#SBATCH --mail-user=miralv@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} 

### Test the model for Pt = 36. 
module purge
module load GCC/5.4.0-2.26
module load CMake/3.5.2
module load OpenMPI/1.10.3

cmake ..
make

###skriver ut n error duration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
	
n=$(echo $((2**14)))
mpirun -np 6 ./poisson_parallel $n $OMP_NUM_THREADS
