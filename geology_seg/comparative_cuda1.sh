for i in 1, 1, 1
do
    python train.py --model 'ccnet_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'pspnet_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1

done

for i in 1, 1
do
    python train.py --model 'deeplabv3plus_mobilenet' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'deeplabv3plus_xception' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'deeplabv3plus_resnet101' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'pspnet_resnet101' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'pspnet_mobilenet' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    python train.py --model 'pspnet_xception' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1

done

python train.py --model 'deeplabv3plus_resnet50' --batch_size 8 --loss_type 'cross_entropy' --crop_size 256 --lr 1e-5 --gen_dataset no --fine_grained coarse --data pos  --pretrained_backbone no --no 3 --additional cmp --device cuda:1
    