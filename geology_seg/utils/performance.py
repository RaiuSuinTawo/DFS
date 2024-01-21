#from cProfile import label
from cProfile import label
from http.client import MOVED_PERMANENTLY
import torch



def PA(outputs, labels):
    if not outputs.shape == labels.shape and len(label.shape) != 1:
        raise ValueError('the shape of outputs is '+str(outputs.shape)+' and the shape of labels is '+str(labels.shape))
    return float((outputs.type(labels.dtype) == labels).type(labels.dtype).sum()/outputs.numel())



def mPA(outputs, labels):
    if not outputs.shape == labels.shape:
        raise ValueError('the shape of outputs is '+str(outputs.shape)+' and the shape of labels is '+str(labels.shape))
    a = 0.0
    if len(labels.shape)<3:

        for l in labels.unique():
            mask = (l==labels)
            #print((outputs*mask).type(torch.bool).type(labels.dtype).sum(dim=1).sum(dim=1))
            #print(mask.sum(dim=1).sum(dim=1))
            a += (outputs[mask].type(labels.dtype)==labels[mask]).type(labels.dtype).sum()/mask.type(labels.dtype).sum()
        
        return float((a/len(labels.unique())))
    else:
        for o, l in zip(outputs, labels):
            a += mPA(o, l)
        #print(a)
        return a


def iou(outputs, labels, label):
    region_l = (label==labels).type(labels.dtype)
    region_o = (label==outputs).type(labels.dtype)
    if len(outputs.shape)<3:
        return (region_l*region_o).type(torch.bool).type(labels.dtype).sum()/(region_l+region_o).type(torch.bool).type(labels.dtype).sum()


    return (region_l*region_o).type(torch.bool).type(labels.dtype).sum(dim=1).sum(dim=1)/\
            (region_l+region_o).type(torch.bool).type(labels.dtype).sum(dim=1).sum(dim=1)



def mIOU(outputs, labels):
    if not outputs.shape == labels.shape:
        raise ValueError('the shape of outputs is '+str(outputs.shape)+' and the shape of labels is '+str(labels.shape))
    miou = 0.0
    if len(outputs.shape)<3:
        for l in labels.unique():
            miou += iou(outputs, labels, l)
        return float(miou/len(labels.unique()))
    else:
        for o, l in zip(outputs, labels):
            miou += mIOU(o,l)
        return miou





def f1Score(outputs, labels):
    pass


def Confusion_Mat(outputs, labels):
    if not outputs.shape == labels.shape:
        raise ValueError('the shape of outputs is '+str(outputs.shape)+' and the shape of labels is '+str(labels.shape))