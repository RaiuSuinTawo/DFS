from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, VanillaHead
from ._ccnet import RCCAModule, PSPModule
from .segformer import SegFormer
from .new_arch import SegFormer_mod as seg_mod
from .setr import SETRModel
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)
from .dual_modal_arch import MyNet

import torchfcn
from model.unet import UNet

from .resnet_add import resnet_dual as rd_add
from .resnet_am import resnet_dual as rd_am

try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url



def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    elif name=='ccnet':
        return_layers = {'stage4': 'out'}
        classifier = RCCAModule(inplanes, 512, num_classes)
    elif name=='pspnet':
        return_layers = {'stage4': 'out'}
        classifier = PSPModule(inplanes, num_classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048 if int(backbone_name[6:]) > 34 else 512
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        
        
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    elif name=='ccnet':
        return_layers = {'layer4': 'out'}
        classifier = RCCAModule(inplanes, 512, num_classes)
    elif name=='pspnet':
        return_layers = {'layer4': 'out'}
        classifier = PSPModule(inplanes, num_classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    elif name=='ccnet':
        return_layers = {'conv4': 'out'}
        classifier = RCCAModule(inplanes, 512, num_classes)
    elif name=='pspnet':
        return_layers = {'conv4': 'out'}
        classifier = PSPModule(inplanes, num_classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    elif name=='ccnet':
        return_layers = {'high_level_features': 'out'}
        classifier = RCCAModule(inplanes, 512, num_classes)
    elif name=='pspnet':
        return_layers = {'high_level_features': 'out'}
        classifier = PSPModule(inplanes, num_classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model
    
def _load_danet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 0:
        output_channel = 128
    elif output_stride == 1:
        output_channel = 256
    elif output_stride == 2:
        output_channel = 512
    else:
        raise ValueError('expected input output_stride as danet setting, but got illegal input as '+str(output_stride)+'.')
    if name == 'danet_add':
        return_layers = {'layer4': 'out'}
        backbone = rd_add(output_channel)
        if backbone == 'ccnet':
            head = RCCAModule(in_channels=output_channel, out_channels=output_channel, num_classes=num_classes)
        elif backbone == 'pspnet':
            head = PSPModule(output_channel, num_classes)
        else:
            head = VanillaHead(output_channel, num_classes)
    elif name == 'danet_am':
        return_layers = {'layer4': 'out'}
        backbone = rd_am(output_channel)
        if backbone == 'ccnet':
            head = RCCAModule(in_channels=output_channel, out_channels=output_channel, num_classes=num_classes)
        elif backbone == 'pspnet':
            head = PSPModule(output_channel, num_classes)
        else:
            head = VanillaHead(output_channel, num_classes)
    return DeepLabV3(backbone, head)

def _load_mynet(name, backbone_name, num_classes, output_stride, pretrained_backbone, **kwargs):
    if 'dims'  not in kwargs:
        kwargs['dims'] = 48
    if 'inchans'  not in kwargs:
        kwargs['inchans'] = [3,3]
    if 'layers' not in kwargs:
        kwargs['layers'] = [2,2,12,2]
    #if kwargs['inchans'] is not None:
    #    inchans = kwargs['inchans']
    if 'recurrence' not in kwargs:
        kwargs['recurrence'] = 2
    if 'window_size' not in kwargs:
        kwargs['window_size'] = 8
    if 'drop_rate' not in kwargs:
        kwargs['drop_rate'] = 0.1
    return MyNet(num_classes=num_classes, attention=name.split('_')[1], **kwargs)

def _segm_vanilla(name, backbone_name, num_classes, output_stride, pretrained_backbone, **kwargs):
    if name.split('_')[0] == 'danet':
        return _load_danet(name, backbone_name, num_classes, output_stride, pretrained_backbone)
    elif 'segformer' in name:
        return SegFormer(num_classes=num_classes, phi=name.split('_')[1], pretrained=pretrained_backbone)
    elif name == 'setr':
        return SETRModel(patch_size=(16, 16), 
                    in_channels=3, 
                    out_channels=num_classes, 
                    hidden_size=1024, 
                    num_hidden_layers=6, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])
    elif name.split('_')[0] == 'segmod':
        return seg_mod(num_classes=num_classes, phi=name.split('_')[1], pretrained=pretrained_backbone)
    elif name.split('_')[0] == 'mynet':
        return _load_mynet(name, backbone_name, num_classes, output_stride, pretrained_backbone, **kwargs)
    #backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    #return DeepLabV3(backbone, head) # backbone: Res34
#========================================================================================================
#========================================================================================================
#========================================================================================================


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs):
    if backbone == 'vanilla':
        return _segm_vanilla(arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs)
    elif backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone=='unet':
        model = UNet(3, n_classes=num_classes)
    elif backbone=='fcn8s':
        model = torchfcn.models.FCN8s(n_class=num_classes)
    elif backbone=='fcn16s':
        model = torchfcn.models.FCN16s(n_class=num_classes)
    elif backbone=='fcn32s':
        model = torchfcn.models.FCN32s(n_class=num_classes) 
    else:
        raise NotImplementedError
    return model



#=======================================================================================================================
#==============================MODELS ENTRANCE==========================================================================
#=======================================================================================================================



# fcn
def fcn32s(num_classes=21, output_stride=None, pretrained_backbone=None, **kwargs):
    return _load_model("fcn32s", "fcn32s", num_classes, output_stride, pretrained_backbone)

def fcn16s(num_classes=21, output_stride=None, pretrained_backbone=None, **kwargs):
    return _load_model("fcn16s", "fcn16s", num_classes, output_stride, pretrained_backbone)

def fcn8s(num_classes=21, output_stride=None, pretrained_backbone=None, **kwargs):
    return _load_model("fcn8s", "fcn8s", num_classes, output_stride, pretrained_backbone)


# unet
def unet(num_classes=21, output_stride=None, pretrained_backbone=None, **kwargs):
    return _load_model("unet", "unet", num_classes, output_stride, pretrained_backbone)


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a Xception backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# CCNet
def ccnet_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ccnet_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ccnet_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ccnet_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ccnet_hrnetv2_32(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'hrnetv2_32', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ccnet_hrnetv2_48(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ccnet', 'hrnetv2_48', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

#danet
def danet_add(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_add', 'vanilla', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def danet_am(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_am', 'vanilla', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def danet_add_ccnet(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_add', 'ccnet', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def danet_am_ccnet(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_am', 'ccnet', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def danet_add_pspnet(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_add', 'pspnet', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def danet_am_pspnet(num_classes=21, output_stride=0, pretrained_backbone=True, **kwargs):
    return _load_model('danet_am', 'pspnet', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


#PSPNet
def pspnet_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def pspnet_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def pspnet_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def pspnet_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def pspnet_hrnetv2_32(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'hrnetv2_32', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def pspnet_hrnetv2_48(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('pspnet', 'hrnetv2_48', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

#DANet/by wc gsp
def danet_add_vanilla(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('danetadd', 'vanilla', num_classes, output_stride=0, pretrained_backbone=pretrained_backbone)

def danet_am_vanilla(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('danetam', 'vanilla', num_classes, output_stride=0, pretrained_backbone=pretrained_backbone)

# SegFormers
def segformer_b0(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segformer_b1(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segformer_b2(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segformer_b3(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segformer_b4(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segformer_b5(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segformer_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

# SETR
def setr(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('setr', 'vanilla', num_classes, output_stride, pretrained_backbone)

# Segformer MODIFIED
def segmod_b0(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b0', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segmod_b1(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b1', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segmod_b2(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b2', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segmod_b3(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b3', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segmod_b4(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b4', 'vanilla', num_classes, output_stride, pretrained_backbone)

def segmod_b5(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('segmod_b5', 'vanilla', num_classes, output_stride, pretrained_backbone)

#========================================================================
#                           mynet                                      
#========================================================================
def mynet_criss(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('mynet_criss', 'vanilla', num_classes, output_stride, pretrained_backbone, **kwargs)

def mynet_swin(num_classes=21, output_stride=None, pretrained_backbone=True, **kwargs):
    return _load_model('mynet_swin', 'vanilla', num_classes, output_stride, pretrained_backbone, **kwargs)

if __name__ == '__main__':
    segformer_b0()
    print('B0 SUC')
    segformer_b1()
    print('B1 SUC')
    segformer_b2()
    print('B2 SUC')
    segformer_b3()
    print('B3 SUC')
    segformer_b4()
    print('B4 SUC')
    segformer_b5()
    print('B5 SUC')
