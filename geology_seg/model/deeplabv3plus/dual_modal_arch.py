#from builtins import print
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
#import dual_modal_utils as util
from .dual_modal_utils import Encoder
#from .backbone.SF_backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
#from .dual-modal_utils import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

        

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2) # b, hw, c
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[48, 96, 192, 384], embedding_dim=768, dropout_ratio=0.1, **kwargs):
        super(SegFormerHead, self).__init__()
        if 'abl' in kwargs and kwargs['abl'] is not None and \
                kwargs['abl'].split('_')[0]=='ablation':
            self.abl = kwargs['abl']
            
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.decode_abl = True
        self.count = 0
        for c in ['c1', 'c2', 'c3', 'c4']:
            if c in self.abl:
                self.count += 1
        if self.count == 0:
            self.count = 4
            self.decode_abl = False
            
        self.linear_fuse = ConvModule(
            c1=embedding_dim*self.count,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        fused = []
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        if not self.decode_abl or (self.decode_abl and 'c4' in self.abl):
            _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            fused.append(_c4)
        if not self.decode_abl or (self.decode_abl and 'c3' in self.abl): 
            _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            fused.append(_c3)
        if not self.decode_abl or (self.decode_abl and 'c2' in self.abl): 
            _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            fused.append(_c2)
        if not self.decode_abl or (self.decode_abl and 'c1' in self.abl): 
            _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
            fused.append(_c1)
        
        _c = self.linear_fuse(torch.cat(fused, dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class MyNet(nn.Module):
    def __init__(self, num_classes = 5, attention='swin', **kwargs):
        super(MyNet, self).__init__()
        dim = kwargs['dims'] if 'dims' in kwargs else 48
        dims = []
        for i in range(4):
            dims.append(dim)
            dim *= 2
        self.encoder = Encoder(num_classes=num_classes, attention=attention, **kwargs)
        self.decode_head = SegFormerHead(num_classes, in_channels=dims, embedding_dim=768, dropout_ratio=0.1, **kwargs)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        pos = inputs[:,:3,:,:]
        neg = inputs[:,3:,:,:]
        #print(pos.shape, neg.shape)
        x = self.encoder.forward(pos, neg)
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
