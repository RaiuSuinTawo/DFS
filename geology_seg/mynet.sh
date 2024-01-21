#扩大中间层数
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,27,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 扩大中间层数

#减少中间层数 [采用]
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional ablation_pairSoftmax

#

#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional ablation_CrossSwin

#到nyu-depth试试看
#python train_on_rgbd.py --model mynet_swin --batch_size 4 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 到nyu-depth试试看

#! /bin/bash
#read -p "the times to experiment: " n
for i in 1, 1, 1, 1, 1
do
    python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
    --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
    --epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional ablation_Compensate_CrossSwin_c1
    python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
    --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
    --epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional ablation_Compensate_CrossSwin_c1_c2
    python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
    --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
    --epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional ablation_Compensate_CrossSwin_c1_c2_c3
done




#===========================以上两个用了warmup和cosine anneal，不满足对照需要重新训练
#不用recurrence
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,18,2 --recurrence 1 --window_size 8 --heads 4 --head_dim 8 --additional 不用recurrence

#增多recurrence
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,18,2 --recurrence 3 --window_size 8 --heads 4 --head_dim 8 --additional 增多recurrence

#增大drop
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --drop_rate 0.2 --additional 增大drop

#减小head
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 2 --head_dim 16 --drop_rate 0.1 --additional 减小head

#recurrence->4
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 4 --window_size 8 --heads 4 --head_dim 8 --additional recurrence_4

#到nyu-depth试试看
#python train_on_rgbd.py --model mynet_swin --batch_size 4 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 2 --window_size 8 --heads 4 --head_dim 8 --additional 到nyu-depth试试看

#recurrence->4
#python train.py --model mynet_swin --batch_size 8 --loss_type 'cross_entropy' \
#--crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both \
#--epoch 500 --layers 2,2,12,2 --recurrence 4 --window_size 8 --heads 4 --head_dim 8 --additional recurrence_4



#下一步尝试使用overlap patch embedding