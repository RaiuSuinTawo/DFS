import os
import json
import base64
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import argparse
from multiprocessing import shared_memory as shm
import multiprocessing as mp

label_dic_fined = {"_background_": 0,
             "Cht": 1,
             "Lmp": 2,
             "Lms": 3,
             "Lsc": 4,
             "Lss": 5,
             "Lim": 6,
             "Lvf": 7,
             "Lvm": 8,
             "Lml": 9,
             "Lii": 10,
             "Lsm": 11,
             "Others": 12,
             "Qm": 13,
             "Qp": 14,
             "P": 15,
             "P1": 16,
             "K": 17,
             "K1": 18
             }

label_dic_coarse = {"_background_": 0,
             "Cht": 1,
             "Lmp": 1,
             "Lms": 1,
             "Lsc": 1,
             "Lss": 1,
             "Lim": 1,
             "Lvf": 1,
             "Lvm": 1,
             "Lml": 1,
             "Lii": 1,
             "Lsm": 1,
             "Others": 2,
             "Qm": 3,
             "Qp": 3,
             "P": 4,
             "P1": 4,
             "K": 4,
             "K1": 4
             }

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../geo_seg",help="path to Dataset")                   
    parser.add_argument("--shm_name", type=str, default='default' ,help="name of the allocated shared memory")   
    parser.add_argument("--mode", type=str, default='pos' ,help="read pos image or neg or both")       
    parser.add_argument("--no", type=str, default=0 ,help="experiment group") 
    #parser.add_argument("--is_Train", type=str, default='yes')     
    parser.add_argument("--is_fined", type=str, default='yes')                    
    
    return parser

def json_to_label(json_path, is_fined):
    """将json文件转换为标签图像"""
    with open(json_path) as f:
        dic = json.load(f)
        labels = dic["labels"]
        image_data = dic["image_data"]
    label_to_class = {}
    label_dic = label_dic_fined if is_fined else label_dic_coarse
    for i, label_name in enumerate(labels):
        label_to_class[i] = label_dic[label_name.split(":")[0].strip()]
    # print(label_to_class)
    image = base64.b64decode(image_data)
    img = Image.open(BytesIO(image))
    # img.show()
    PILToTensor = transforms.PILToTensor()
    img = PILToTensor(img).to(torch.int64).squeeze(dim=0)
    h, w =img.shape
    img_cls = torch.zeros(h, w)
    for key, value in label_to_class.items():
        img_cls[img == key] = value
    return img_cls.unsqueeze(dim=0)

def read_images(img_dir, mode="pos", is_train=True, is_fined=True):
    """读取所有数据集图像并标注"""
    features, labels = [], []
    features_b = []
    file_pos_dir = os.path.join(img_dir, "train" if is_train else "val", "Images", "pos")
    file_neg_dir = os.path.join(img_dir, "train" if is_train else "val", "Images", "neg")
    label_dir = os.path.join(img_dir, "train" if is_train else "val", "SegClasses")
    file_list, label_list = [], []
    file_list_b = []
    if mode == "both":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_pos_dir, name.split(".")[0] + "+.jpg"))
                file_list_b.append(os.path.join(file_neg_dir, name.split(".")[0] + "-.jpg"))
    elif mode == "pos":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_pos_dir, name.split(".")[0] + "+.jpg"))
    elif mode == "neg":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_neg_dir, name.split(".")[0] + "-.jpg"))

    assert len(file_list) == len(label_list)

    PILToTensor = transforms.PILToTensor()

    for i in range(len(file_list)):
        features.append(PILToTensor(Image.open(file_list[i])))
        if mode == 'both':
            features[-1] = torch.cat([features[-1], PILToTensor(Image.open(file_list_b[i]))], dim=0).tolist()
        labels.append(json_to_label(label_list[i], is_fined)).tolist()

    return [features, labels]

def main():
    opts = get_argparser().parse_args()
    is_fined = True if opts.is_fined =='yes' else False
    shm = shm.ShareableList([read_images(opts.data_dir, opts.mode, is_train=True, is_fined=is_fined), 
                             read_images(opts.data_dir, opts.mode, is_train=False, is_fined=is_fined)],
                            name=opts.shm_name)
    print('shm allocate to ['+opts.shm_name+'] is done!')