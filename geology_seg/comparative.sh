#for i in 1, 1
#do
#python train.py --model 'ccnet_mobilenet' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'ccnet_resnet101' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'ccnet_xception' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'deeplabv3_mobilenet' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'deeplabv3_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'deeplabv3_xception' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'deeplabv3_resnet101' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#done

#python train.py --model 'segformer_b0' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp

#python train.py --model 'segformer_b1' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'segformer_b4' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'segformer_b5' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp
#python train.py --model 'pspnet_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python train.py --model 'deeplabv3plus_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python send_mail.py

#python train.py --model 'segformer_b2' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python train.py --model 'segformer_b3' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python send_mail.py 

#python train.py --model 'unet' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python train.py --model 'setr' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python train.py --model 'fcn8s' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no
#python send_mail.py 

#python train.py --model 'segmod_b0' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
#python train.py --model 'segmod_b1' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
#python train.py --model 'segmod_b2' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
#python train.py --model 'segmod_b3' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
#python train.py --model 'segmod_b4' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
#python train.py --model 'segmod_b5' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data both --pretrained_backbone no --no 2
for i in 1, 1
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
