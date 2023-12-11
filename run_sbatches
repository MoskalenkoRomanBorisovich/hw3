#! /bin/bash

mkdir -p bemchmark_results

for n_tasks in 1 2 4; do
    sbatch -N 1 --ntasks-per-node=$n_tasks --output="bemchmark_results/out_${n_tasks}.csv" slurm_run.sh
done

for n_tasks in 2 4 6; do
    k=$(($n_tasks * 4))
    echo $k
    sbatch -N 4 --ntasks-per-node=$n_tasks --output="bemchmark_results/out_${k}.csv" slurm_run.sh
done
