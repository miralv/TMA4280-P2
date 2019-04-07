#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1	### Number of tasks (MPI Processes)
#SBATCH --cpus-per-task=1	### Number of threads per task (OMP threads)
#SBATCH --job-name="test"
#SBATCH --output="omp_zeta_1.out"

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} 

module purge
module load GCC/5.4.0-2.26
module load CMake/3.5.2
module load OpenMPI/1.10.3

cmake ..
make

###skriver ut n error duration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
	
for k in {3..14}
do
	n=$(echo $((2**$k)))
	srun ./zeta3 $n $OMP_NUM_THREADS
done

