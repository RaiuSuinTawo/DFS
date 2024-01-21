import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import random
import time

def rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
    feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
    
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


class RGBDDataset(Dataset):
    def __init__(self, indexes, crop_size, data_dir):
        begin = time.time()
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_d = torchvision.transforms.Normalize(
            mean=[0.255,0.255,0.255], std=[0.157,0.157,0.1570])
        #2.2983583213818437
        self.crop_size = crop_size
        dataset = h5py.File(data_dir)
        keys = list(dataset.keys())
        idx_img, idx_depth, idx_label = keys.index('images'), keys.index('depths'), keys.index('labels')
        features, features_b, labels = torch.from_numpy(np.array(dataset[keys[idx_img]]).astype(np.int16))[indexes], \
                    torch.from_numpy(np.array(dataset[keys[idx_depth]]).astype(np.int16))[indexes].unsqueeze(dim=1).repeat(1,3,1,1),\
                    torch.from_numpy(np.array(dataset[keys[idx_label]]).astype(np.int16))[indexes]
        #print(features.shape)
        # print(features[0].shape, labels[0].shape)
        self.features = [self.normalize_image_rgb(feature)
                         for feature in self.filter(features)]  # pos和neg应该相同尺寸，所以filter返回值相同
        self.features_b = [self.normalize_image_depth(feature)
                            for feature in self.filter(features_b)]
        self.labels = self.filter(labels)
        print('read ' + str(len(self.features)) + ' examples')
        during = int(time.time()-begin)
        print('='*20+' spent {} minutes {} seconds '.format(int(during/60), during%60)+'='*20)

    def normalize_image_rgb(self, img):
        return self.transform(img.float() / 255.)
    
    def normalize_image_depth(self, img):
        return self.transform_d(img.float() / 9.)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[-2] >= self.crop_size[0] and
                img.shape[-1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        
        feature, feature_b, label = rand_crop_pair(self.features[idx], self.features_b[idx], self.labels[idx], *self.crop_size)
        feature = torch.cat([feature, feature_b], dim=0)
        return feature, label

    def __len__(self):
        return len(self.features)


def load_data_rgbd(batch_size, crop_size, data_dir, train_por=0.5):
    """加载地质语义分割数据集"""
    # limited 1449
    index_of_dataset = list(range(1449))
    random.shuffle(index_of_dataset)
    train_len = int(1449 * train_por)
    train_indexes = index_of_dataset[:train_len]
    valid_indexes = index_of_dataset[train_len:]
    num_workers = 0
    dataset = h5py.File(data_dir)
    keys = list(dataset.keys())
    num_classes = dataset[keys[keys.index('names')]].shape[1]
    train_iter = torch.utils.data.DataLoader(
        RGBDDataset(train_indexes, crop_size,  data_dir=data_dir), 
        batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
    val_iter = torch.utils.data.DataLoader(
        RGBDDataset(valid_indexes, crop_size, data_dir=data_dir), 
        batch_size, drop_last=True, num_workers=num_workers, pin_memory=False)
    return train_iter, val_iter, num_classes


if __name__ == "__main__":
    pass