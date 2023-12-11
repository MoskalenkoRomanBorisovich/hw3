#! /bin/bash

#SBATCH --time=01:00:00

#SBATCH --constraint="type_a"
#SBATCH --cpus-per-task=1

for N in 10000 25000 50000; do
    mpirun ./a.out $N 1000000 0.0001
done
