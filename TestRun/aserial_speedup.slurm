#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --mem=23000 
#SBATCH --ntasks-per-node=1	### Number of tasks (MPI Processes)
#SBATCH --cpus-per-task=1	### Number of threads per task (OMP threads)
#SBATCH --job-name="speedup_test"
#SBATCH --output="bserial_speedup.out"
#SBATCH --mail-user=miralv@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} 

 
cmake ..
make

###skriver ut n error duration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
	
n=$(echo $((2**14)))
./poisson n
