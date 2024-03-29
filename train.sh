#!/usr/bin/env bash

#SBATCH --job-name=lab2
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=coms030144
#SBATCH --gres=gpu:1
#SBATCH --time=0:20:00
#SBATCH --mem=4GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"
# achieve 85.50 with adamW and lr=0.003  loss=0.15866

# Please note, if you want run this script, please have a look at the README.md file
# to determine the modifications on instructions. Here are just examples.

python train.py --learning-rate 0.003 --sgd-momentum 0.9
