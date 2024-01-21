python predict.py --input ../../pred_images --label ../../labels --model pspnet_resnet101 --weights_path ./checkpoints/pspnet_resnet101 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1

python predict.py --input ../../pred_images --label ../../labels --model pspnet_mobilenet --weights_path ./checkpoints/pspnet_mobilenet \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1

python predict.py --input ../../pred_images --label ../../labels --model pspnet_xception --weights_path ./checkpoints/pspnet_xception \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1

python predict.py --input ../../pred_images --label ../../labels --model ccnet_resnet50 --weights_path ./checkpoints/ccnet_resnet50 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1

python predict.py --input ../../pred_images --label ../../labels --model ccnet_mobilenet --weights_path ./checkpoints/ccnet_mobilenet \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1

python predict.py --input ../../pred_images --label ../../labels --model ccnet_xception --weights_path ./checkpoints/ccnet_xception \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_pspnet_resnet101 --data pos --no 1
