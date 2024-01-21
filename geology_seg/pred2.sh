
python predict.py --input ../../pred_images --label ../../labels --model unet --weights_path ./checkpoints/unet \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_unet --data pos --no 2

python predict.py --input ../../pred_images --label ../../labels --model segformer_b0 --weights_path ./checkpoints/segformer_b0 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_segformer_b0 --data pos --no 2

python predict.py --input ../../pred_images --label ../../labels --model segformer_b1 --weights_path ./checkpoints/segformer_b1 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_segformer_b1 --data pos --no 2

python predict.py --input ../../pred_images --label ../../labels --model segformer_b2 --weights_path ./checkpoints/segformer_b2 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_segformer_b2 --data pos --no 2

python predict.py --input ../../pred_images --label ../../labels --model segformer_b3 --weights_path ./checkpoints/segformer_b3 \
--additional ablation_Compensate_CrossSwin --crop_size 256,256 --save_results_to ../../save_preds_segformer_b3 --data pos --no 2

