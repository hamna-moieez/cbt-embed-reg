#!/bin/bash
#SBATCH -J cbtERRun
#SBATCH -o cbt.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:4
#SBATCH -c 4
#SBATCH --mem 500000MB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 
#SBATCH --mail-type=fail
#SBATCH --mail-user=hamna21797@gmail.com

module load slurm
module load cudnn8.0-cuda11.1/8.0.4.30
python -u main_cl.py ../us3dData/train --epochs 1000 --workers 3 --batch-size 128 --EWC_task_count 0 --checkpoint-dir ./checkpoints
