#$ -S /bin/bash

#$ -q short
#$ -l ngpus=1
#$ -l ncpus=10
#$ -l h_vmem=32G
#$ -l h_rt=11:00:00

source ~/.bashrc
conda activate /mmfs1/storage/users/jiangz9/pytorch/
sleep 2
cd /mmfs1/home/users/jiangz9/Code/AP_Prune

VAR=""
for ARG in "$*";do
	VAR+="${ARG} "
done
echo $VAR


python train.py $VAR