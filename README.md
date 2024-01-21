# Dual-input Fusion Segmenter: an effective framework for semantic segmentation of detritus from river sands

## **Models Available for Training**

Our repository includes the DFS model and various semantic segmentation models used for comparative experiments, such as fcn, unet, ccnet, deeplab, segformer, and more (refer to geologyseg/model and geology_seg/model/deeplabv3plus). Additionally, we explore different backbones for semantic segmentation, including resnet, hrnet, xception, etc. (found in geology_seg/model/deeplabv3plus/backbone).

For a detailed view of the code for our DFS model, please see geology_seg/model/deeplabv3plus/new_arch.py and geology_seg/model/deeplabv3plus/new_utils.py. The code for model training is available in geology_seg/train.py, while the code for model prediction can be found in geology_seg/predict.py.

## **Experimental Data**

The experimental data used in the paper is located in .txt files with val and pred prefixes under geology_seg and val_texts, along with a .csv file under geology_seg/results_coarse.

## **Remaining**

Kindly utilize the pre-existing scripts in the repository to retrain the model and execute the required tasks.
