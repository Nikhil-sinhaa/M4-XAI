#!/bin/bash
#SBATCH -A iitdh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4-00:00:00
#SBATCH -J m4-cnn_small
#SBATCH -o logs/%x-%j.out
source scripts/slurm/common.sh
vrun vit_small
vrun vgg16
vrun resnet101
vrun resnet152
