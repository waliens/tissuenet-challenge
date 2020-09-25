#!/bin/bash
#
#SBATCH --job-name=eval_model
#SBATCH --output=eval_model.log
#SBATCH --partition=debug
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

/home/rmormont/miniconda3/envs/py-gpu/bin/python eval_model_on_slides.py --device "cuda:0" --n_jobs 5 -m "$SCRATCH/tissuenet/metadata" -i "$SCRATCH/tissuenet/wsis"