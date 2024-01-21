#from ctypes import util
import os
import argparse
import copy
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import dataset
from dataset.dataset import load_data_geo
from dataset.dataset2 import load_data_geo as load_data_geo_training
from model import deeplabv3, unet, fcn
from model.deeplabv3plus import modeling
from utils.loss import FocalLoss, ohem_loss
from torch.utils.data import DataLoader
import pandas as pd
import dataset.gen_dataset
import utils
import utils.performance
import math

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import smtplib
from email.mime.text import MIMEText
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_argparser():
    parser = argparse.ArgumentParser()
    available_models = sorted(name for name in modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--data_dir", type=str, default="../../geo_seg",
                        help="path to Dataset")                    
    parser.add_argument("--output_stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=300,
                        help="epoch number (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    #parser.add_argument("--train_por", type=float, default=0.5)

    parser.add_argument("--gen_dataset", type=str,choices=['yes','no'],default='yes')

    parser.add_argument("--fine_grained", type=str,choices=['fined','coarse'],default='fine')
    parser.add_argument("--data", type=str,choices=['neg','pos', 'both'],default='pos')
    parser.add_argument("--read", type=str,choices=['pre','train'],default='pre')
    parser.add_argument("--pretrained_backbone", '-p', type=str,choices=['yes','no'],default='yes')
    parser.add_argument("--num_workers", type=int,default=0)
    
    parser.add_argument("--layers", type=str,default='2,2,6,2')
    parser.add_argument("--inchans", type=str,default='3,3') # 使用配置文件导入，研究一下
    parser.add_argument("--emb_dim", type=int,default=48)
    parser.add_argument("--recurrence", type=int,default=2)
    parser.add_argument("--window_size", type=int,default=8)
    parser.add_argument("--heads", type=int,default=8)
    parser.add_argument("--head_dim", type=int,default=6)
    parser.add_argument("--drop_rate", type=float,default=0.1)
    
    parser.add_argument("--additional", type=str,default=None)
    
    parser.add_argument("--no", type=int,default=0)
    
    
    
    return parser


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    y = y.squeeze(dim=1)
    if y.shape == y_hat.shape:
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())
    else:
        raise ValueError('WRONG IN OUTPUT AND LABELS SIZE.')


def send_mail(msg, title, sender='719751595@qq.com', accepters=['719751595@qq.com', '554872480@qq.com']):
    message = MIMEText(msg,'plain','utf-8')
    message['Subject'] = title
    message['From'] = sender
    
    smtpObj = smtplib.SMTP_SSL('smtp.qq.com')
    #smtpObj.connect('smtp.smail.nju.edu.cn')
    smtpObj.login(sender.split('@')[0], 'iirgrkewstuubdde')
    for a in accepters:
        message['To'] = a
        smtpObj.sendmail(sender, a, message.as_string())
    smtpObj.quit()
    
def str2list(s):
    if type(s) is list:
        return s
    items = ''.join(s.strip()).split(',')
    ret = []
    for i in items:
        ret.append(int(i))
    return ret

def mannual_seed(seed):
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed) #random and transforms
    torch.backends.cudnn.deterministic = True # cudnn

def main():
    #try:
        opts = get_argparser().parse_args()
        if opts.seed is not None:
            mannual_seed(opts.seed)
        #opts.output_stride <<= 1
        if opts.gen_dataset=='yes':
            dataset.gen_dataset.gen(4,4,2) #train:val:pred = 4:4:2
        else:
            print('NO GENERATE NEW TRAIN_SET AND VALID_SET!')

        if opts.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(opts.device)
        print("Device: %s" % device)
        #try:
            #send_mail(str(opts), 'Success')
            #raise
        opts.layers = str2list(opts.layers)    
        opts.inchans = str2list(opts.inchans)
        #print(opts.layers)
        #raise 
        if opts.additional.split('_')[0] == 'ablation':
            abl = opts.additional
        else:
            abl = None
        model = modeling.__dict__[opts.model](num_classes = 19 if opts.fine_grained=='fined' else 5, 
                                            output_stride=opts.output_stride, 
                                            pretrained_backbone = True if opts.pretrained_backbone=='yes' else False,
                                            layers = opts.layers,
                                            dims=opts.emb_dim,
                                            inchans=opts.inchans,
                                            recurrence=opts.recurrence,
                                            window_size=opts.window_size,
                                            heads = opts.heads,
                                            head_dim=opts.head_dim,
                                            drop_rate=opts.drop_rate,
                                            abl=abl)
        if opts.read == 'pre':
            train_loader, val_loader = load_data_geo(opts.batch_size, (opts.crop_size, opts.crop_size), opts.data_dir, \
                                                    True if opts.fine_grained=='fined' else False, mode=opts.data)
        else:
            train_loader, val_loader = load_data_geo_training(opts.batch_size, (opts.crop_size, opts.crop_size), opts.data_dir, \
                                                    True if opts.fine_grained=='fined' else False, mode=opts.data)
        # model = net
        # model = getattr(deeplabv3, 'resnet50')(
        #     pretrained=True,
        #     num_classes=17
        # )
        # model = unet.UNet(3, n_classes=17)
        model.to(device)
        def CELoss(inputs, targets):
            return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

        FCLoss = FocalLoss(alpha=0.25, gamma=2)

        # lr
        optimizer = Adam(model.parameters(), lr=opts.lr)
        warm_up_iter = 20
        T_max = 50	# 周期
        lr_max = opts.lr	# 最大值
        lr_min = opts.lr*0.01	# 最小值
        lr_lambda = lambda cur_iter: (cur_iter+1)/warm_up_iter if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max)*math.pi)))/opts.lr
 
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_miou = 0.
        best_pa, best_mpa = 0. ,0.
        model.train()
        train_results = {'loss':[], 'PA':[], 'mPA':[], 'mIOU':[]}
        valid_results = {'PA':[], 'mPA':[], 'mIOU':[]}

        #mpa_table = dict()


        save_pre = '{}_{}_{}_{}_{}_{}'.format(opts.model, opts.epoch, opts.lr, opts.batch_size, opts.crop_size, opts.additional)
        train_metric = Accumulator(5)
        val_metric = Accumulator(4)
        val_index = []
        
        for i in range(1, opts.epoch+1):
            #print("===========第{}轮训练开始===========".format(i+1))
            torch.cuda.empty_cache()
            train_metric.reset()
            train_bar = tqdm(train_loader)
            for batch in train_bar:
                data, targets = batch
                data = data.to(device)
                targets = targets.to(device)
                #del batch
                #print(data.shape)
                #print(target.shape)
                outputs = model(data)
                if opts.loss_type == "cross_entropy":
                    loss = CELoss(outputs, targets.squeeze(dim=1).long())
                else:
                    loss = FCLoss(outputs, targets.squeeze(dim=1).long())

                #print(outputs.shape)
                #print(targets.shape)
                #raise ValueError
                #train_acc = accuracy(outputs, targets)
                

                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()

                pred = outputs.argmax(dim=1)
                train_metric.add(targets.shape[0], loss.sum(), utils.performance.PA(pred, targets.squeeze(dim=1))*targets.shape[0], \
                                                                utils.performance.mPA(pred, targets.squeeze(dim=1)), \
                                                                utils.performance.mIOU(pred, targets.squeeze(dim=1)))
                #print(f'mpa: {train_metric[3] / train_metric[0]:.3f}',f'miou: {train_metric[4] / train_metric[0]:.3f}')
                train_bar.set_description('Train Epoch: [{}/{}] Avg Loss: {:.3f}'.format(i, opts.epoch, train_metric[0] / train_metric[1]))
            #print('current lr:', str(lr_scheduler.get_lr()))
            lr_scheduler.step()
            #raise
            train_results['loss'].append(train_metric[1]/train_metric[0])
            train_results['PA'].append(train_metric[2]/train_metric[0])
            train_results['mPA'].append(train_metric[3]/train_metric[0])
            train_results['mIOU'].append(train_metric[4]/train_metric[0])
            '''
            if train_results['loss'][-1] < 0.1:
                lr_max = lr_max // 10
                lr_min //= 10
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                            lambda cur_iter: (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max)*math.pi)))/opts.lr, 
                            last_epoch=i-1)
            '''
            
            
            data_frame = pd.DataFrame(data=train_results, index=range(1, i + 1))
            to = 'results_' + opts.fine_grained
            if not os.path.exists(to):
                os.mkdir(to)
            data_frame.to_csv(to+'/{}_statistics.csv'.format(save_pre), index_label='epoch')
            
            #if i%2==0:
            val_index.append(i)
            model.eval()
            val_metric.reset()
            with torch.no_grad():
                
                for batch in val_loader:
                    data, targets = batch
                    data = data.to(device)
                    targets = targets.squeeze(dim=1).long().to(device)
                    output = model(data)

                    output = output.argmax(dim=1).squeeze(dim=1)
                    val_metric.add(targets.shape[0], utils.performance.PA(output, targets)*targets.shape[0], \
                                                        utils.performance.mPA(output, targets), \
                                                            utils.performance.mIOU(output, targets))
            
            valid_results['PA'].append(val_metric[1]/val_metric[0])
            valid_results['mPA'].append(val_metric[2]/val_metric[0])
            valid_results['mIOU'].append(val_metric[3]/val_metric[0])


            val_miou = valid_results['mIOU'][-1]
            if val_miou > best_miou:
                best_miou = val_miou
                best_pa = valid_results['PA'][-1]
                best_mpa = valid_results['mPA'][-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                
                
            data_frame = pd.DataFrame(data=valid_results, index=val_index)
            if not os.path.exists(to):
                os.mkdir(to)
            data_frame.to_csv(to+'/val_{}_statistics.csv'.format(save_pre), index_label='epoch')
            print(f'epoch {i} val mIOU: {val_miou:.3f}')
    
        # record
        name = opts.model if opts.additional is None else ' '.join([opts.model, opts.additional])
        settings = str(opts)
        performance = f'best PA is {best_pa:.3f}, \nbest mpa is {best_mpa:.3f}, \nand best miou is {best_miou:.3f}.\n'
        with open('./val'+str(opts.no)+'.txt', 'a+') as f:
            f.write('='*10+'Model'+'='*10+\
                '\n'+name+'\n'+\
                    '='*10+'Settings'+'='*10+\
                    '\n'+settings+'\n'+\
                        '='*10+'Performance'+'='*10+\
                            '\n'+performance+'\n')
        
        # save
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        save_path = os.path.join('./checkpoints', opts.model)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_name_intern = f'{opts.fine_grained}_{opts.data}_{best_pa:.3f}_{best_mpa:.3f}_{best_miou:.3f}_'
        model.load_state_dict(best_model_wts)
        
        models = os.listdir(save_path)
        flag = False
        for m in models:
            if opts.model in m:
                if float(m.split('_')[-2]) < best_miou:
                    torch.save(model.state_dict(), 
                            os.path.join(save_path, model_name_intern+"weight.pth"))
                flag = True
        if not flag:
            torch.save(model.state_dict(), 
                    os.path.join(save_path, model_name_intern+"weight.pth"))
        #send_mail(str(opts)+'\nBEST PERFORMANCE: PA={}, mPA={}, mIOU={}'.format(best_pa, best_mpa, best_miou), 'Success')
'''
    except:
        if os.path.exists('./val'+str(opts.no)+'.txt'):
            with open('./val'+str(opts.no)+'.txt', 'a+') as f:
                f.write('='*10+'Model'+'='*10+\
                    '\n'+name+'\n'+\
                        '='*10+'Settings'+'='*10+\
                            '\n'+settings+'\n'+\
                                'Failed......'+'\n')
    '''

if __name__ == "__main__":
    main()
