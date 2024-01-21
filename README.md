# Dual-input Fusion Segmenter: an effective framework for semantic segmentation of detritus from river sands

## Models available for training

Our DFS model and other semantic segmentation models (for comparative experiments), including fcn, unet, ccnet, deeplab, segformer, etc. (see geologyseg/model, geology_seg/model/deeplabv3plus), as well as different backbone for semantic segmentation, including resnet, hrnet, xception, etc. (see geology_seg/model/deeplabv3plus/backbone).

Specifically, you can see the code for our DFS model in geology_seg/model/deeplabv3plus/new_arch.py and geology_seg/model/deeplabv3plus/new_utils.py.

The code for model training in geology_seg/train.py, and the code for model prediction in geology_seg/predict.py.

## Model parameters

geology_seg/checkpoints contains the trained parameters of most of the aforementioned models (DFS is called mynet_swin here).

## Experimental data

The .txt file with val and pred prefixes under geology_seg and val_texts, and the .csv file under geology_seg/results_coarse save the experimental data used in the paper

## Remaining

You can use some existing scripts in the repository to complete retraining or directly load the parameters we have trained into the model to apply the model to complete the tasks you need to perform.
