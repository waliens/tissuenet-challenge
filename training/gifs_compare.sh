#!/bin/sh
#
#SBATCH --job-name=compare_result
#SBATCH --output=compare_log.txt
#
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00
#SBATCH --mem=30G
#SBATCH --partition=debug

/home/rmormont/miniconda3/envs/py-gpu/bin/python monuseg_one_plot_param_score.py