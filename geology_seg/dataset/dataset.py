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
            features_b.append(PILToTensor(Image.open(file_list_b[i])))
        labels.append(json_to_label(label_list[i], is_fined))

    if mode == 'both':
        return features, features_b, labels
    else:
        return features, labels


# img_dir = "/data1/fyc/dataset/geo_seg"
# f, l = read_images(img_dir)
# print(f[0].shape, l[0].shape)


def rand_crop(feature, label, mode, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
    feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
    
def rand_crop_pair(feature, feature_b, label, mode, height, width):
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
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.mode = mode
        self.size = batch_sz
        if mode == 'both':
            features, features_b, labels = read_images(data_dir, mode=mode, is_train=is_train, is_fined=is_fined)
        else:
            features, labels = read_images(data_dir, mode=mode, is_train=is_train, is_fined=is_fined)
        # print(features[0].shape, labels[0].shape)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]  # pos和neg应该相同尺寸，所以filter返回值相同
        if mode == 'both':
            self.features_b = [self.normalize_image(feature)
                            for feature in self.filter(features_b)]
        self.labels = self.filter(labels)
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255.)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        
        if self.mode == 'both':
            feature, feature_b, label = rand_crop_pair(self.features[idx], self.features_b[idx], self.labels[idx], self.mode, *self.crop_size)
            feature = torch.cat([feature, feature_b], dim=0)
        else:
            feature, label = rand_crop(self.features[idx], self.labels[idx], self.mode, *self.crop_size)
        return feature, label

    def __len__(self):
        return len(self.features)


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
    import random
    import time
    from multiprocessing import shared_memory as shm
    import multiprocessing as mp
    file_pos_dir = os.path.join('../../../geo_seg', "train" , "Images", 'pos')
    
    def test(sl):
        a = sl[10]
        im = Image.open(BytesIO(base64.b64decode(a)))
        print('读取')
        print('共享列表大小为：'+str(len(sl)))
        
    begin = time.time()
    d = []
    PILToTensor = transforms.PILToTensor()
    for f in os.listdir(file_pos_dir):
        with open(os.path.join(file_pos_dir, f)) as f:
            b64 = base64.b64encode(f.read()).decode()
            d.append(b64)
        #print('open images plus one')
    print('存放至共享内存')
    shm_a = shm.ShareableList(d,name='1')
    print('分配完成')
    print('分配使用了 '+str(time()-begin)+' seconds')
    begin = time.time()
    print('a[100]='+str(shm_a[100]))
    p = mp.Process(target=test, args=[shm_a,])
    p.start()
    p.join()
    print('进程回收')
    print('回收使用了 '+str(time()-begin)+' seconds')
    shm_a.shm.close()
    shm_a.shm.unlink()