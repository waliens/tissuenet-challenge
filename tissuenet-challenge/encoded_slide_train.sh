#!/bin/bash
#
#SBATCH --job-name=train_encoded
#SBATCH --output=train_encoded.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --partition=debug

/home/rmormont/miniconda3/envs/py-gpu/bin/python encoded_slide_train.py \
  --random_seed 42 \
  --base_path "/scratch/users/rmormont/tissuenet" \
  --epochs 30 \
  --batch_size 4 \
  --train_size 0.8 \
  --learning_rate 0.01 \
  --device "cuda:0" \
  --n_jobs 4