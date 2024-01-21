import os
import base64
import json
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize
import argparse
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from glob import glob
from model import deeplabv3
from model.deeplabv3plus import modeling
from dataset.dataset import json_to_label as load_label
import utils.performance as performace
#from dataset.dataset import rand_crop, json_to_label, label_dic_coarse
#from torch.nn import DataParallel as DP
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--label", type=str, default=None, help="path to the label from pred_image")
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        help='model name')
    parser.add_argument("--weights_path", type=str, default='./checkpoints',
                        help="path to load model weights")
    
    parser.add_argument("--output_stride", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--crop_size", type=str, default='256,256')
    parser.add_argument("--fine_grained", type=str,choices=['fined','coarse'],default='coarse')
    parser.add_argument("--data", type=str,choices=['both','pos'],default='both')
    
    parser.add_argument("--save_results_to", type=str, default='./save_images',
                        help="path to save_results")
    
    parser.add_argument("--layers", type=str,default='2,2,12,2')
    parser.add_argument("--inchans", type=str,default='3,3') # 使用配置文件导入，研究一下
    parser.add_argument("--emb_dim", type=int,default=48)
    parser.add_argument("--recurrence", type=int,default=2)
    parser.add_argument("--window_size", type=int,default=8)
    parser.add_argument("--heads", type=int,default=4)
    parser.add_argument("--head_dim", type=int,default=8)
    parser.add_argument("--additional", type=str,default=None)
    
    parser.add_argument("--no", type=str,default=0)
    parser.add_argument("--device", type=str,default='cuda:0')

    return parser

def str2list(s):
    if type(s) is list:
        return s
    items = ''.join(s.strip()).split(',')
    ret = []
    for i in items:
        ret.append(int(i))
    return ret

def predict(img_path, device, net, crop_size=[256,256], window_size=8): # split the proto image, predict them indepently
    PILToTensor = torchvision.transforms.PILToTensor()
    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if len(img_path)==2:
        img, img_ = PILToTensor(Image.open(img_path[0])), PILToTensor(Image.open(img_path[1]))
        #print(img.shape)
        #print(img_.shape)
        X = torch.cat([transform(img.float() / 255.).unsqueeze(0), transform(img_.float() / 255.).unsqueeze(0)], dim=1).to(device)
    else:
        img = PILToTensor(Image.open(img_path))
        X = transform(img.float() / 255.).unsqueeze(0).to(device)
    b, c, H, W = X.shape # cut it to 256*256 to predict
    
    #n_w_H, n_w_W = H//(window_size*32), W//(window_size*32)
    #crop_size = [n_w_H*(window_size*8), n_w_W*(window_size*8)]
    #print(crop_size)
    
    n_H, n_W = H // crop_size[0], W // crop_size[1]
    pred = torch.zeros(1, H, W)
    
    assert crop_size[0]%window_size==0 and crop_size[1]%window_size==0
    
    for i in range(n_H):
        for j in range(n_W):
            #print(X[:, :, i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]].shape)
            pred[:, i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]] = \
                net(X[:, :, i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]]).argmax(dim=1)
    if H % crop_size[0] != 0 or W % crop_size[1] != 0:
        pred[:, H-crop_size[0]:H, W-crop_size[1]:W] = net(X[:, :, H-crop_size[0]:H, W-crop_size[1]:W]).argmax(dim=1)
    
    return pred

geo_COLORMAP = [[0, 0, 0], [128, 0, 0], [128, 128, 0], [0, 128, 0],  [0, 0, 128]]

label_dic = {'background':0, 'L':1, 'Others':2, 'Qt':3, 'F':4}

def cam_mask(mask,palette,n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[mask[:, :]==c] = palette[c]
    seg_img = seg_img.astype('uint8')
        #per_class_metric[c] = [performace.PA()]
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

def decode_and_save(pred, save_path, device):
    pic = cam_mask(pred.squeeze(0).cpu().numpy(), geo_COLORMAP, 5)
    # pic = Image.fromarray(pic)
    pic.save(save_path+".png")
    del pic
    
    img_cls = pred.squeeze(0)
    h, w = img_cls.shape
    img = torch.zeros(h, w)

    labels = []
    for key, value in label_dic.items():
        if value in img_cls:
            labels.append(key)
    labels.sort(key=lambda x: x in labels)
    for label in labels:
        img[img_cls==label_dic[label]] = labels.index(label)
    img = Image.fromarray(img.numpy()).convert("RGB")
    # img.save("."+save_path.split(".")[1]+".jpg")
    img_buffer = BytesIO()
    img.save(img_buffer, format="png")
    img_data = base64.b64encode(img_buffer.getvalue())
    
    dic = dict()
    dic["labels"] = labels
    dic["image_data"] = img_data.decode()
    json.dump(dic, open(save_path+'.json', 'w'), indent=4)

def metric(pred, label, conf_mat): #返回这张图象各类别的pa, iou指标
    for i in range(len(label_dic)):
        positive = (pred==i)
        negative = (pred!=i)
        true = (label==i)
        false = (label!=i)
        conf_mat[i][0] += (positive * true).sum() # tp
        conf_mat[i][1] += (negative * false).sum() # fn
        conf_mat[i][2] += (positive * false).sum() # fp
        conf_mat[i][3] += (negative * true).sum() # tn
    return conf_mat


def main():
    opts = get_argparser().parse_args()
    if opts.model == 'mynet_swin':
        t = geo_COLORMAP[1]
        geo_COLORMAP[1] = geo_COLORMAP[0]
        geo_COLORMAP[0] = t
    
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    print('Pred Start')
    # 加载数据
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    

    # 加载模型
    opts.layers = str2list(opts.layers)    
    opts.inchans = str2list(opts.inchans)
    opts.crop_size = str2list(opts.crop_size)
    #print(opts.layers)
    #raise 
    if opts.additional.split('_')[0] == 'ablation':
        abl = opts.additional
    else:
        abl = None
        
    #model = modeling.__dict__[opts.model](num_classes=opts.num_classes, 
    #                                      output_stride=opts.output_stride, 
    #                                      pretrained_backbone=False),[0,1]
    model = modeling.__dict__[opts.model](num_classes = 19 if opts.fine_grained=='fined' else 5, 
                                            output_stride=opts.output_stride, 
                                            pretrained_backbone = False,
                                            layers = opts.layers,
                                            dims=opts.emb_dim,
                                            inchans=opts.inchans,
                                            recurrence=opts.recurrence,
                                            window_size=opts.window_size,
                                            heads = opts.heads,
                                            head_dim=opts.head_dim,
                                            abl=abl)
    #model = model.module
    if os.path.isdir(opts.weights_path):
        opts.weights_path = os.path.join(opts.weights_path, os.listdir(opts.weights_path)[0])
    param_path = opts.weights_path
    model.load_state_dict(torch.load(param_path))
    #model = model.module
    model.to(device)
    if opts.save_results_to is not None:
        os.makedirs(opts.save_results_to, exist_ok=True)
    predicted = [] # 同名只预测一次
    conf_mat = dict() # 混淆矩阵
    for i in range(len(label_dic)):
        conf_mat[i] = torch.zeros(4).numpy() # tp, fn, fp, tn
        
        
    with torch.no_grad():
        model = model.eval()
        pred_bar = tqdm(image_files)
        for img_path in pred_bar:
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            pred = None
            if opts.data == 'both':
                #print(image_files)
                if img_name[:-1] not in predicted:
                    if (img_name[-1]=='+' and os.path.join(os.path.dirname(img_path), img_name[:-1]+'-.'+ext) in image_files) or \
                        (img_name[-1]=='-' and os.path.join(os.path.dirname(img_path), img_name[:-1]+'+.'+ext) in image_files):
                        #要求图片后缀+-要一致
                        pred = predict([os.path.join(os.path.dirname(img_path), img_name[:-1]+'-.'+ext), 
                                        os.path.join(os.path.dirname(img_path), img_name[:-1]+'+.'+ext)], device, model, opts.crop_size)
                        predicted.append(img_name[:-1])
                        #print('predict image '+img_name[:-1]+' Succesfully!')
                        decode_and_save(pred, os.path.join(opts.save_results_to, img_name[:-1]), device)
                        #print('save predicted image '+img_name[:-1]+' Succesfully!')
                        
                    else:
                        raise ValueError('the image pair is not both exist in the input file.')
                else:
                    #print('the image pair has been predicted.')
                    pass
            elif img_name[-1]=='+' and opts.data=='pos':
                pred = predict(img_path, device, model, opts.crop_size) # 此时img文件下需要是pos
                predicted.append(img_name[:-1])
                #print('predict image '+img_name[:-1]+' Succesfully!')
                decode_and_save(pred, os.path.join(opts.save_results_to, img_name[:-1]), device)
                #print('save predicted image '+img_name[:-1]+' Succesfully!')
            
            elif img_name[-1]=='-' and opts.data=='neg':
                pred = predict(img_path, device, model, opts.crop_size) # 此时img文件下需要是neg
                predicted.append(img_name[:-1])
                #print('predict image '+img_name[:-1]+' Succesfully!')
                decode_and_save(pred, os.path.join(opts.save_results_to, img_name[:-1]), device)
                #print('save predicted image '+img_name[:-1]+' Succesfully!')
            
            if opts.label is not None and pred is not None:
                if img_name[:-1]+'.json' in os.listdir(opts.label):
                    metric(pred, load_label(os.path.join(opts.label, img_name[:-1]+'.json'), is_fined=False), conf_mat=conf_mat)
                else:
                    raise ValueError('there is not label in the label-path param.')
            
        np.save('./conf_mat'+opts.model, conf_mat, allow_pickle=True)
        '''
        with open('./pred'+str(opts.no)+'.txt', 'a+') as f:
            #f.write('pred '+opts.model+' successfully!\n')
            if opts.label is not None:
                keys = list(label_dic.keys())
                for i in range(len(label_dic)):
                    pa = (conf_mat[i][0]+conf_mat[i][1])/(conf_mat[i].sum())
                    miou = conf_mat[i][0]/(conf_mat[i][2]+conf_mat[i][3])
                    description = 'with matter to '+keys[i]+' in '+opts.model+', PA is {:.3f}, and mIOU is {:.3f},\nin the confusion matrix, '.format(pa, miou)+ \
                            'TP is {}, FN is {}, FP is {}, and TN is {}.\n'.format(conf_mat[i][0], conf_mat[i][1], conf_mat[i][2], conf_mat[i][3])
                    print(description)
                    f.write(description)
        '''
if __name__ == '__main__':
    #try:
        main()
    #except:
    #    with open('./temp_pred.txt', 'a+') as f:
    #        f.write('pred failed!\n')
            

