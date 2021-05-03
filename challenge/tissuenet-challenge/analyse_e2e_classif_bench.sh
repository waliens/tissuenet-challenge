#!/bin/bash
#
#SBATCH --job-name=analyse_e2e_classif_benchmark
#SBATCH --output=analyse_e2e_classif_benchmark.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

/home/rmormont/miniconda3/envs/py-gpu/bin/python analyse_e2e_classif_benchmark.py