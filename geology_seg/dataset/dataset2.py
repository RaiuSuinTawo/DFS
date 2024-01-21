# 我们希望将读数据的操作放在每个batch getitem中，这样虽然会让训练速度变慢，但是能节省内存，方便多卡训练

from asyncore import file_dispatcher
import os
import json
import base64
from re import L
from statistics import mode
from unicodedata import name
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import random

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


# json_path = r"/data1/fyc/dataset/geo_seg"
# img = json_to_label(json_path)
# print(img[105:115, 130:140])

'''
def read_images(img_dir, mode="pos", is_train=True, is_fined=True):
    """读取所有数据集图像并标注"""
    
    if mode == "both":
        
    elif mode == "pos":
        
    elif mode == "neg":
        

    assert len(file_list) == len(label_list)
    
    PILToTensor = transforms.PILToTensor()

    for i in range(len(file_list)):
        features.append(PILToTensor(Image.open(file_list[i])))
        if mode == 'both':
            features_b.append(PILToTensor(Image.open(file_list_b[i])))
        labels.append(json_to_label(label_list[i], is_fined))
    
    if mode == 'both':
        return file_list, file_list_b, label_list
    else:
        return file_list, label_list
'''

# img_dir = "/data1/fyc/dataset/geo_seg"
# f, l = read_images(img_dir)
# print(f[0].shape, l[0].shape)


def rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def rand_crops(features, labels, height, width):
    """随机裁剪特征和标签图像"""
    retf = []
    retl = []
    for feature, label in zip(features, labels):
        rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
        feature = torchvision.transforms.functional.crop(feature, *rect)
        label = torchvision.transforms.functional.crop(label, *rect)
        retf.append(feature)
        retl.append(label)
    return retf, retl
    
def rand_crop_pair(feature, feature_b, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    feature_b = torchvision.transforms.functional.crop(feature_b, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, feature_b, label


# f_c, l_c = rand_crop(f[0], l[0], 2000, 3000)
# plt.subplot(121)
# plt.imshow(f_c.permute(1,2,0))
# plt.subplot(122)
# plt.imshow(l_c)
# plt.show()


class GeoSegDataset(Dataset):
    def __init__(self, is_train, crop_size, mode, data_dir, is_fined, batch_sz):
        #print(data_dir)
        self.is_fined = is_fined
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.mode = mode
        self.file_pos_dir = os.path.join(data_dir, "train" if is_train else "val", "Images", "pos")
        self.file_neg_dir = os.path.join(data_dir, "train" if is_train else "val", "Images", "neg")
        self.label_dir = os.path.join(data_dir, "train" if is_train else "val", "SegClasses")
        self.size = batch_sz
        self.labels = os.listdir(self.label_dir)
        random.shuffle(self.labels)
        
        PILToTensor = transforms.PILToTensor()
        '''
        if mode == 'both':
            self.features, self.features_b, self.labels = os.listdir(self.file_neg_dir), os.listdir(self.file_pos_dir), os.listdir(self.label_dir)
        elif mode == 'pos':
            self.features, self.labels, self.file_dir = os.listdir(self.file_pos_dir), os.listdir(self.label_dir), self.file_pos_dir
        elif mode == 'neg':
            self.features, self.labels, self.file_dir = os.listdir(self.file_neg_dir), os.listdir(self.label_dir), self.file_neg_dir
        else:
            raise NotImplementedError
        # print(features[0].shape, labels[0].shape)
        
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]  # pos和neg应该相同尺寸，所以filter返回值相同
        '''
        # [self.normalize_image(feature) for feature in self.filter(features_b)]
        # self.filter(labels)
        self.names = self.labels[:4*self.size]
        self.label_bank = [json_to_label(os.path.join(self.label_dir, l), self.is_fined) for l in self.names]
        self.img_bank = []
        if mode == 'pos':
            for name in self.names:
                self.img_bank.append(PILToTensor(Image.open(os.path.join(self.file_pos_dir, \
                                        name.split('.')[0]+'+.jpg'))))
        elif mode == 'neg':
            for name in self.names:
                self.img_bank.append(PILToTensor(Image.open(os.path.join(self.file_neg_dir, \
                                        name.split('.')[0]+'-.jpg'))))
        elif mode == 'both':
            for name in self.names:
                self.img_bank.append(torch.cat([PILToTensor(Image.open(os.path.join(self.file_pos_dir, \
                                        name.split('.')[0]+'+.jpg'))), \
                                            PILToTensor(Image.open(os.path.join(self.file_neg_dir, \
                                        name.split('.')[0]+'-.jpg')))], dim=0))
            
                
        else:
            raise NotImplementedError
        self.label_bank, self.img_bank = self.filter(self.label_bank), self.filter(self.img_bank)
        self.img_bank, self.label_bank = rand_crops(self.img_bank, self.label_bank, *self.crop_size)
        self.img_bank = self.normalize_images(self.img_bank)
        print('Train' if is_train else 'Val','Dataset Prepared!')

    def normalize_image(self, img):
        return self.transform(img.float() / 255.)
    def normalize_images(self, imgs):
        if self.mode == 'both':
            return [torch.cat([self.transform(img[0:3, :, :] / 255.), self.transform(img[3:, :, :] / 255.)], dim=0) for img in imgs]
        return [self.transform(img.float() / 255.) for img in imgs]
    
    
    def filter(self, imgs):
        #for im in imgs:
        #    print(im.shape)
        ret = []
        for img in imgs:
            if (img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1]):
                ret.append(img)
        return ret
    
    def getd(self, name):
        id = self.names.index(name)
        feature, label = self.img_bank[id], self.label_bank[id]
        del self.names[id]
        del self.img_bank[id]
        del self.label_bank[id]
        return feature, label
        
    
    def __getitem__(self, idx):
        PILToTensor = transforms.PILToTensor()
        
        if self.labels[idx] not in self.names:
            f, l = [], []
            
            for i in range(idx, idx+self.size*4):
                p = torch.tensor([])
                if i >= len(self.labels):
                    break
                file_path = []
                if self.mode == 'pos':
                    file_path.append(os.path.join(self.file_pos_dir, self.labels[i].split('.')[0]+'+.jpg'))
                elif self.mode == 'neg':
                    file_path.append(os.path.join(self.file_neg_dir, self.labels[i].split('.')[0]+'-.jpg'))
                elif self.mode == 'both':
                    file_path.append(os.path.join(self.file_neg_dir, self.labels[i].split('.')[0]+'-.jpg'))
                    file_path.append(os.path.join(self.file_pos_dir, self.labels[i].split('.')[0]+'+.jpg'))
                    
                
                #print(file_path)
                for file in file_path:
                    p = torch.cat([p, PILToTensor(Image.open(file))], dim=0)
                    #print(p.shape)
                    #print(file)
                if len(self.filter([p])) == 0:
                        continue
                    
                l.append(json_to_label(os.path.join(self.label_dir, self.labels[i]), 
                                                     self.is_fined))
                f.append(p)
                self.names.append(self.labels[i])
                        
                        
            #f, l = self.filter(f), self.filter(l)
            f, l = rand_crops(f, l, *self.crop_size)
            f = self.normalize_images(f)
            
            self.label_bank.extend(l) 
            self.img_bank.extend(f) 
            del l
            del f
        return self.getd(self.labels[idx])
            
        '''
        feature, label = rand_crop(self.features[idx], self.labels[idx],
                                    *self.crop_size)
        if self.mode == 'both':
            feature_b, _ = rand_crop(self.features_b[idx], self.labels[idx],*self.crop_size)
            feature = torch.cat([feature, feature_b], dim=0)
        return feature, label
        '''
        

    def __len__(self):
        return len(self.labels)


def load_data_geo(batch_size, crop_size, data_dir, is_fined=True, mode='neg'):
    """加载地质语义分割数据集"""
    #mode = "neg"
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(
        GeoSegDataset(True, crop_size, mode=mode, data_dir=data_dir, is_fined=is_fined, batch_sz=batch_size), 
        batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
    val_iter = torch.utils.data.DataLoader(
        GeoSegDataset(False, crop_size, mode=mode, data_dir=data_dir, is_fined=is_fined, batch_sz=batch_size), 
        batch_size, drop_last=True, num_workers=num_workers, pin_memory=False)
    return train_iter, val_iter


if __name__ == "__main__":
    train, val = load_data_geo(8,(224,224),'../geo_seg')
    for i in range(10):
        for batch in train:
            data, target = batch
            print(str(i+1),'DATA',data.shape)
            print(str(i+1),'LABEL',target.shape)
        for batch in val:
            data, target = batch
            print(str(i+1),'DATA',data.shape)
            print(str(i+1),'LABEL',target.shape)
    