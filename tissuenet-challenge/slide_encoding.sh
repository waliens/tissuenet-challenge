#!/bin/bash
#
#SBATCH --job-name=slide_encoding
#SBATCH --output=slide_encoding.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

/home/rmormont/miniconda3/envs/py-gpu/bin/python slide_encoding.py