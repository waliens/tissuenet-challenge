#!/bin/bash
#
#SBATCH --job-name=slide_encoding
#SBATCH --output=slide_encoding.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --partition=debug

/home/rmormont/miniconda3/envs/py-gpu/bin/python slide_encoding.py