import enum
import os
import argparse
import random
import shutil
#def maked(path):
#    if not os.path.exists(os.path.dirname(path)):
#        maked()

def makedir(*args):
    for d in args:
        if not os.path.exists(os.path.dirname(d)):
            makedir(os.path.dirname(d))
        os.mkdir(d)


def gen(train, valid, pred):

    indexes = []
    pos_source = '../../images/pos'
    neg_source = '../../images/neg'
    label_source = '../../labels'
    pos_train_tar = '../../geo_seg/train/Images/pos'
    neg_train_tar = '../../geo_seg/train/Images/neg'
    pos_valid_tar = '../../geo_seg/val/Images/pos'
    neg_valid_tar = '../../geo_seg/val/Images/neg'
    pos_pred_tar = '../../geo_seg/val/Images/pos'
    neg_pred_tar = '../../geo_seg/val/Images/neg'
    label_train_tar = '../../geo_seg/train/SegClasses'
    label_valid_tar = '../../geo_seg/val/SegClasses'
    label_pred_tar = '../../geo_seg/val/SegClasses'

    pos = os.listdir(pos_source)
    neg = os.listdir(neg_source)
    for l in os.listdir(label_source):
        cmp = l[:-5]
        if cmp+'+.jpg' in pos and cmp+'-.jpg' in neg:
            indexes.append(cmp)
    train_len = int(len(indexes) * (train/(train+valid+pred)))
    valid_len = int(len(indexes) * (valid/(train+valid+pred)))
    random.shuffle(indexes)
    
    train_index = indexes[: train_len]
    valid_index = indexes[train_len: valid_len]
    pred_index = indexes[valid_len: ]

    if os.path.exists('../../geo_seg'):
        shutil.rmtree('../../geo_seg')
    makedir(pos_train_tar, pos_valid_tar, pos_pred_tar, neg_train_tar, neg_valid_tar, neg_pred_tar, label_train_tar, label_valid_tar, label_pred_tar)
    
    for idx in train_index:
        shutil.copy(os.path.join(pos_source, idx+'+.jpg'), pos_train_tar)
        shutil.copy(os.path.join(neg_source, idx+'-.jpg'), neg_train_tar)
        shutil.copy(os.path.join(label_source, idx+'.json'), label_train_tar)
    print('COPY TRAIN DATA COMPLISHED!')
    
    for idx in valid_index:
        shutil.copy(os.path.join(pos_source, idx+'+.jpg'), pos_valid_tar)
        shutil.copy(os.path.join(neg_source, idx+'-.jpg'), neg_valid_tar)
        shutil.copy(os.path.join(label_source, idx+'.json'), label_valid_tar)
    print('COPY VALID DATA COMPLISHED!')
    
    for idx in pred_index:
        shutil.copy(os.path.join(pos_source, idx+'+.jpg'), pos_pred_tar)
        shutil.copy(os.path.join(neg_source, idx+'-.jpg'), neg_pred_tar)
        shutil.copy(os.path.join(label_source, idx+'.json'), label_pred_tar)
    print('COPY VALID DATA COMPLISHED!')
    
if __name__ == '__main__':
    pass