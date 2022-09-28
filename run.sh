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
# python -u main_cl.py ../data/cssl/Potsdam/imgs/ --epochs 1000 --workers 3 --batch-size 128 --EWC_task_count 0 --checkpoint-dir ./checkpoints

python -u main_cl.py ../data/cssl/sen12ms/imgs/ --epochs 300 --workers 3 --batch-size 128 --ER_task_count 0 --checkpoint-dir ./checkpoints_sen12ms

# make a new docker container with a single gpu
# docker run --gpus="device=1" -it --rm --name "cbter" -v /raid/data/marsocci:/marsocci -v $HOME=$HOME marsocci/sox1

# make a new docker container with a multiple gpus
# docker run --gpus='"device=0,1"' -it --rm --name "cbter" -v /raid/data/marsocci:/marsocci -v $HOME=$HOME marsocci/sox1


#list of docker containers
# docker ps

#start a preexisting docker container
#docker start -i <container id>