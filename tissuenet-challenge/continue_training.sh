#!/bin/bash
#
#SBATCH --job-name=continue_training
#SBATCH --output=continue_training.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

/home/rmormont/miniconda3/envs/py-gpu/bin/python continue_training.py \
  --start_model "/scratch/users/rmormont/tissuenet/saved_models/densenet121_mtdp_e_59_val_0.7780_sco_0.9423_z2_1602868062.683283.pth" \
  --image_path "/scratch/users/rmormont/tissuenet/patches" \
  --metadata_path "/scratch/users/rmormont/tissuenet/metadata" \
  --model_path "/scratch/users/rmormont/tissuenet/models/continued" \
  --pretrained "mtdp" \
  --architecture "densenet121" \
  --epochs 30 \
  --batch_size 24 \
  --zoom_level 2 \
  --random_seed 653543684 \
  --learning_rate 0.0001 \
  --device "cuda:0" \
  --n_jobs 8 \
  --aug_elastic_alpha_low 9 \
  --aug_elastic_alpha_high 11 \
  --aug_elastic_sigma_low 80 \
  --aug_elastic_sigma_high 120 \
  --aug_hed_bias_range 0.0125 \
  --aug_hed_coef_range 0.1

