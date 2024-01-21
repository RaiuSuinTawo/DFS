python predict.py --input ../../pred_images --label ../../labels --model mynet_swin --weights_path ./checkpoints/mynet_swin \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_mynet_swin

#python predict.py --input ../../pred_images --label ../../labels --model ccnet_resnet101 --weights_path ./checkpoints/ccnet_resnet101 \
#--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_ccnet_resnet101 --data pos

#python predict.py --input ../../pred_images --label ../../labels --model deeplabv3_resnet101 --weights_path ./checkpoints/deeplabv3_resnet101 \
#--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3_resnet101 --data pos

#python predict.py --input ../../pred_images --label ../../labels --model deeplabv3plus_resnet50 --weights_path ./checkpoints/deeplabv3plus_resnet50 \
#--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos

#python predict.py --input ../../pred_images --label ../../labels --model deeplabv3plus_mobilenet --weights_path ./checkpoints/deeplabv3plus_mobilenet \
#--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos

#python predict.py --input ../../pred_images --label ../../labels --model deeplabv3plus_xception --weights_path ./checkpoints/deeplabv3plus_xception \
#--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos

python predict.py --input ../../pred_images --label ../../labels --model deeplabv3_resnet50 --weights_path ./checkpoints/deeplabv3_resnet50  \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos

python predict.py --input ../../pred_images --label ../../labels --model deeplabv3_mobilenet --weights_path ./checkpoints/deeplabv3_mobilenet \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos

python predict.py --input ../../pred_images --label ../../labels --model deeplabv3_xception --weights_path ./checkpoints/deeplabv3_xception \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_deeplabv3plus_resnet50 --data pos
