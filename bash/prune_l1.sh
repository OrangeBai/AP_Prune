#$ -S /bin/bash

#qsub train.sh --dataset cifar10 --net vgg16 --project prune_l1 --name benchmark --num_epoch 120 --train_mode std --lr_scheduler milestones  --batch_size 256  --activation ReLU  --npbar
qsub train.sh --dataset cifar10 --net VGG16 --project prune_l1 --name benchmark --num_epoch 120

for method in 0 2 3 4 5
do
	for amount in 0.4 0.6 0.8 0.95
	do
	qsub train.sh --dataset cifar10 --net vgg16 --project prune_l1 --name l1_${method}_${amount} --num_epoch 120 --amount ${amount} \
        --lr_scheduler milestones --method ${method} --activation ReLU --npbar --amount_setting 0 --skip 5
    done
done
