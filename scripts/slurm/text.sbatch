#!/bin/bash
#SBATCH -A iitdh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4-00:00:00
#SBATCH -J m4-text
#SBATCH -o logs/%x-%j.out
source scripts/slurm/common.sh
for m in distilbert bert_base ernie_2_base roberta_base bert_large; do trun $m; done
