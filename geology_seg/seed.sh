
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 4447 --seed 4447

#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 8527 --seed 8527

#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 508525 --seed 508525

#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 508614 --seed 508614

#! /bin/bash
read -p "the times to experiment: " n
for (( i=0;i<$n;i=i+1 ))
    do
        echo $i
    done



#python send_mail.py

#下一步尝试使用overlap patch embedding