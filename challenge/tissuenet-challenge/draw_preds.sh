#!/bin/bash
#
#SBATCH --job-name=draw_preds
#SBATCH --output=draw_preds.log
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00

/home/rmormont/miniconda3/envs/py-gpu/bin/python analyse_classif_by_hand.py