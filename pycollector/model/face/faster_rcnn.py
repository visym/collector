import numpy as np
import os
import numpy as np
import imp
import torch 
import torchvision.ops
import torch.nn as nn
import torch.nn.functional
import PIL.Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import time
import os
import sys
from math import ceil
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'detection'))


class RpnLayers(nn.Module):
    def __init__(self, weight_file=None):
        super(RpnLayers, self).__init__()
        #global __rpn_layers_weights_dict  # Loaded by MMDNN
        #__rpn_layers_weights_dict = load_weights(weight_file)

        self.rpn_conv_3x3 = self.__conv(2, name='rpn_conv/3x3', in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.rpn_cls_score = self.__conv(2, name='rpn_cls_score', in_channels=512, out_channels=18, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.rpn_bbox_pred = self.__conv(2, name='rpn_bbox_pred', in_channels=512, out_channels=36, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        rpn_conv_3x3_pad = F.pad(x, (1, 1, 1, 1))
        rpn_conv_3x3    = self.rpn_conv_3x3(rpn_conv_3x3_pad)
        rpn_relu_3x3    = F.relu(rpn_conv_3x3)
        rpn_cls_score   = self.rpn_cls_score(rpn_relu_3x3)
        rpn_bbox_pred   = self.rpn_bbox_pred(rpn_relu_3x3)
        return rpn_cls_score, rpn_bbox_pred


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        #layer.state_dict()['weight'].copy_(torch.from_numpy(__rpn_layers_weights_dict[name]['weights']))
        #if 'bias' in __rpn_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__rpn_layers_weights_dict[name]['bias']))
        return layer




class BottomLayers(nn.Module):    
    def __init__(self, weight_file=None):
        super(BottomLayers, self).__init__()
        #global __bottom_layers_weights_dict  # loaded by MMDNN
        #__bottom_layers_weights_dict = load_weights(weight_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=False)
        self.bn_conv1 = self.__batch_normalization(2, 'bn_conv1', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2a_branch2a = self.__conv(2, name='res2a_branch2a', in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.res2a_branch1 = self.__conv(2, name='res2a_branch1', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2a = self.__batch_normalization(2, 'bn2a_branch2a', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.bn2a_branch1 = self.__batch_normalization(2, 'bn2a_branch1', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res2a_branch2b = self.__conv(2, name='res2a_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2b = self.__batch_normalization(2, 'bn2a_branch2b', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2a_branch2c = self.__conv(2, name='res2a_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2c = self.__batch_normalization(2, 'bn2a_branch2c', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res2b_branch2a = self.__conv(2, name='res2b_branch2a', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2a = self.__batch_normalization(2, 'bn2b_branch2a', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2b_branch2b = self.__conv(2, name='res2b_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2b = self.__batch_normalization(2, 'bn2b_branch2b', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2b_branch2c = self.__conv(2, name='res2b_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2c = self.__batch_normalization(2, 'bn2b_branch2c', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res2c_branch2a = self.__conv(2, name='res2c_branch2a', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2a = self.__batch_normalization(2, 'bn2c_branch2a', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2c_branch2b = self.__conv(2, name='res2c_branch2b', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2b = self.__batch_normalization(2, 'bn2c_branch2b', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.res2c_branch2c = self.__conv(2, name='res2c_branch2c', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2c = self.__batch_normalization(2, 'bn2c_branch2c', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res3a_branch1 = self.__conv(2, name='res3a_branch1', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res3a_branch2a = self.__conv(2, name='res3a_branch2a', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn3a_branch1 = self.__batch_normalization(2, 'bn3a_branch1', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.bn3a_branch2a = self.__batch_normalization(2, 'bn3a_branch2a', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3a_branch2b = self.__conv(2, name='res3a_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2b = self.__batch_normalization(2, 'bn3a_branch2b', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3a_branch2c = self.__conv(2, name='res3a_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2c = self.__batch_normalization(2, 'bn3a_branch2c', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res3b1_branch2a = self.__conv(2, name='res3b1_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b1_branch2a = self.__batch_normalization(2, 'bn3b1_branch2a', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b1_branch2b = self.__conv(2, name='res3b1_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b1_branch2b = self.__batch_normalization(2, 'bn3b1_branch2b', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b1_branch2c = self.__conv(2, name='res3b1_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b1_branch2c = self.__batch_normalization(2, 'bn3b1_branch2c', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res3b2_branch2a = self.__conv(2, name='res3b2_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b2_branch2a = self.__batch_normalization(2, 'bn3b2_branch2a', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b2_branch2b = self.__conv(2, name='res3b2_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b2_branch2b = self.__batch_normalization(2, 'bn3b2_branch2b', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b2_branch2c = self.__conv(2, name='res3b2_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b2_branch2c = self.__batch_normalization(2, 'bn3b2_branch2c', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res3b3_branch2a = self.__conv(2, name='res3b3_branch2a', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b3_branch2a = self.__batch_normalization(2, 'bn3b3_branch2a', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b3_branch2b = self.__conv(2, name='res3b3_branch2b', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b3_branch2b = self.__batch_normalization(2, 'bn3b3_branch2b', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.res3b3_branch2c = self.__conv(2, name='res3b3_branch2c', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b3_branch2c = self.__batch_normalization(2, 'bn3b3_branch2c', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res4a_branch1 = self.__conv(2, name='res4a_branch1', in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res4a_branch2a = self.__conv(2, name='res4a_branch2a', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn4a_branch1 = self.__batch_normalization(2, 'bn4a_branch1', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.bn4a_branch2a = self.__batch_normalization(2, 'bn4a_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4a_branch2b = self.__conv(2, name='res4a_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2b = self.__batch_normalization(2, 'bn4a_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4a_branch2c = self.__conv(2, name='res4a_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2c = self.__batch_normalization(2, 'bn4a_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b1_branch2a = self.__conv(2, name='res4b1_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b1_branch2a = self.__batch_normalization(2, 'bn4b1_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b1_branch2b = self.__conv(2, name='res4b1_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b1_branch2b = self.__batch_normalization(2, 'bn4b1_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b1_branch2c = self.__conv(2, name='res4b1_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b1_branch2c = self.__batch_normalization(2, 'bn4b1_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b2_branch2a = self.__conv(2, name='res4b2_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b2_branch2a = self.__batch_normalization(2, 'bn4b2_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b2_branch2b = self.__conv(2, name='res4b2_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b2_branch2b = self.__batch_normalization(2, 'bn4b2_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b2_branch2c = self.__conv(2, name='res4b2_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b2_branch2c = self.__batch_normalization(2, 'bn4b2_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b3_branch2a = self.__conv(2, name='res4b3_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b3_branch2a = self.__batch_normalization(2, 'bn4b3_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b3_branch2b = self.__conv(2, name='res4b3_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b3_branch2b = self.__batch_normalization(2, 'bn4b3_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b3_branch2c = self.__conv(2, name='res4b3_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b3_branch2c = self.__batch_normalization(2, 'bn4b3_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b4_branch2a = self.__conv(2, name='res4b4_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b4_branch2a = self.__batch_normalization(2, 'bn4b4_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b4_branch2b = self.__conv(2, name='res4b4_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b4_branch2b = self.__batch_normalization(2, 'bn4b4_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b4_branch2c = self.__conv(2, name='res4b4_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b4_branch2c = self.__batch_normalization(2, 'bn4b4_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b5_branch2a = self.__conv(2, name='res4b5_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b5_branch2a = self.__batch_normalization(2, 'bn4b5_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b5_branch2b = self.__conv(2, name='res4b5_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b5_branch2b = self.__batch_normalization(2, 'bn4b5_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b5_branch2c = self.__conv(2, name='res4b5_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b5_branch2c = self.__batch_normalization(2, 'bn4b5_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b6_branch2a = self.__conv(2, name='res4b6_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b6_branch2a = self.__batch_normalization(2, 'bn4b6_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b6_branch2b = self.__conv(2, name='res4b6_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b6_branch2b = self.__batch_normalization(2, 'bn4b6_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b6_branch2c = self.__conv(2, name='res4b6_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b6_branch2c = self.__batch_normalization(2, 'bn4b6_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b7_branch2a = self.__conv(2, name='res4b7_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b7_branch2a = self.__batch_normalization(2, 'bn4b7_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b7_branch2b = self.__conv(2, name='res4b7_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b7_branch2b = self.__batch_normalization(2, 'bn4b7_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b7_branch2c = self.__conv(2, name='res4b7_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b7_branch2c = self.__batch_normalization(2, 'bn4b7_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b8_branch2a = self.__conv(2, name='res4b8_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b8_branch2a = self.__batch_normalization(2, 'bn4b8_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b8_branch2b = self.__conv(2, name='res4b8_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b8_branch2b = self.__batch_normalization(2, 'bn4b8_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b8_branch2c = self.__conv(2, name='res4b8_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b8_branch2c = self.__batch_normalization(2, 'bn4b8_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b9_branch2a = self.__conv(2, name='res4b9_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b9_branch2a = self.__batch_normalization(2, 'bn4b9_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b9_branch2b = self.__conv(2, name='res4b9_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b9_branch2b = self.__batch_normalization(2, 'bn4b9_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b9_branch2c = self.__conv(2, name='res4b9_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b9_branch2c = self.__batch_normalization(2, 'bn4b9_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b10_branch2a = self.__conv(2, name='res4b10_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b10_branch2a = self.__batch_normalization(2, 'bn4b10_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b10_branch2b = self.__conv(2, name='res4b10_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b10_branch2b = self.__batch_normalization(2, 'bn4b10_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b10_branch2c = self.__conv(2, name='res4b10_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b10_branch2c = self.__batch_normalization(2, 'bn4b10_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b11_branch2a = self.__conv(2, name='res4b11_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b11_branch2a = self.__batch_normalization(2, 'bn4b11_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b11_branch2b = self.__conv(2, name='res4b11_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b11_branch2b = self.__batch_normalization(2, 'bn4b11_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b11_branch2c = self.__conv(2, name='res4b11_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b11_branch2c = self.__batch_normalization(2, 'bn4b11_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b12_branch2a = self.__conv(2, name='res4b12_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b12_branch2a = self.__batch_normalization(2, 'bn4b12_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b12_branch2b = self.__conv(2, name='res4b12_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b12_branch2b = self.__batch_normalization(2, 'bn4b12_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b12_branch2c = self.__conv(2, name='res4b12_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b12_branch2c = self.__batch_normalization(2, 'bn4b12_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b13_branch2a = self.__conv(2, name='res4b13_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b13_branch2a = self.__batch_normalization(2, 'bn4b13_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b13_branch2b = self.__conv(2, name='res4b13_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b13_branch2b = self.__batch_normalization(2, 'bn4b13_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b13_branch2c = self.__conv(2, name='res4b13_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b13_branch2c = self.__batch_normalization(2, 'bn4b13_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b14_branch2a = self.__conv(2, name='res4b14_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b14_branch2a = self.__batch_normalization(2, 'bn4b14_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b14_branch2b = self.__conv(2, name='res4b14_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b14_branch2b = self.__batch_normalization(2, 'bn4b14_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b14_branch2c = self.__conv(2, name='res4b14_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b14_branch2c = self.__batch_normalization(2, 'bn4b14_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b15_branch2a = self.__conv(2, name='res4b15_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b15_branch2a = self.__batch_normalization(2, 'bn4b15_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b15_branch2b = self.__conv(2, name='res4b15_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b15_branch2b = self.__batch_normalization(2, 'bn4b15_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b15_branch2c = self.__conv(2, name='res4b15_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b15_branch2c = self.__batch_normalization(2, 'bn4b15_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b16_branch2a = self.__conv(2, name='res4b16_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b16_branch2a = self.__batch_normalization(2, 'bn4b16_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b16_branch2b = self.__conv(2, name='res4b16_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b16_branch2b = self.__batch_normalization(2, 'bn4b16_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b16_branch2c = self.__conv(2, name='res4b16_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b16_branch2c = self.__batch_normalization(2, 'bn4b16_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b17_branch2a = self.__conv(2, name='res4b17_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b17_branch2a = self.__batch_normalization(2, 'bn4b17_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b17_branch2b = self.__conv(2, name='res4b17_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b17_branch2b = self.__batch_normalization(2, 'bn4b17_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b17_branch2c = self.__conv(2, name='res4b17_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b17_branch2c = self.__batch_normalization(2, 'bn4b17_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b18_branch2a = self.__conv(2, name='res4b18_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b18_branch2a = self.__batch_normalization(2, 'bn4b18_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b18_branch2b = self.__conv(2, name='res4b18_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b18_branch2b = self.__batch_normalization(2, 'bn4b18_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b18_branch2c = self.__conv(2, name='res4b18_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b18_branch2c = self.__batch_normalization(2, 'bn4b18_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b19_branch2a = self.__conv(2, name='res4b19_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b19_branch2a = self.__batch_normalization(2, 'bn4b19_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b19_branch2b = self.__conv(2, name='res4b19_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b19_branch2b = self.__batch_normalization(2, 'bn4b19_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b19_branch2c = self.__conv(2, name='res4b19_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b19_branch2c = self.__batch_normalization(2, 'bn4b19_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b20_branch2a = self.__conv(2, name='res4b20_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b20_branch2a = self.__batch_normalization(2, 'bn4b20_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b20_branch2b = self.__conv(2, name='res4b20_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b20_branch2b = self.__batch_normalization(2, 'bn4b20_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b20_branch2c = self.__conv(2, name='res4b20_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b20_branch2c = self.__batch_normalization(2, 'bn4b20_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b21_branch2a = self.__conv(2, name='res4b21_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b21_branch2a = self.__batch_normalization(2, 'bn4b21_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b21_branch2b = self.__conv(2, name='res4b21_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b21_branch2b = self.__batch_normalization(2, 'bn4b21_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b21_branch2c = self.__conv(2, name='res4b21_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b21_branch2c = self.__batch_normalization(2, 'bn4b21_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.res4b22_branch2a = self.__conv(2, name='res4b22_branch2a', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b22_branch2a = self.__batch_normalization(2, 'bn4b22_branch2a', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b22_branch2b = self.__conv(2, name='res4b22_branch2b', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b22_branch2b = self.__batch_normalization(2, 'bn4b22_branch2b', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.res4b22_branch2c = self.__conv(2, name='res4b22_branch2c', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b22_branch2c = self.__batch_normalization(2, 'bn4b22_branch2c', num_features=1024, eps=9.99999974738e-06, momentum=0.0)

    def forward(self, x):
        conv1_pad       = F.pad(x, (3, 3, 3, 3))
        conv1           = self.conv1(conv1_pad)
        bn_conv1        = self.bn_conv1(conv1)
        conv1_relu      = F.relu(bn_conv1)
        pool1_pad       = F.pad(conv1_relu, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        res2a_branch2a  = self.res2a_branch2a(pool1)
        res2a_branch1   = self.res2a_branch1(pool1)
        bn2a_branch2a   = self.bn2a_branch2a(res2a_branch2a)
        bn2a_branch1    = self.bn2a_branch1(res2a_branch1)
        res2a_branch2a_relu = F.relu(bn2a_branch2a)
        res2a_branch2b_pad = F.pad(res2a_branch2a_relu, (1, 1, 1, 1))
        res2a_branch2b  = self.res2a_branch2b(res2a_branch2b_pad)
        bn2a_branch2b   = self.bn2a_branch2b(res2a_branch2b)
        res2a_branch2b_relu = F.relu(bn2a_branch2b)
        res2a_branch2c  = self.res2a_branch2c(res2a_branch2b_relu)
        bn2a_branch2c   = self.bn2a_branch2c(res2a_branch2c)
        res2a           = bn2a_branch1 + bn2a_branch2c
        res2a_relu      = F.relu(res2a)
        res2b_branch2a  = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a   = self.bn2b_branch2a(res2b_branch2a)
        res2b_branch2a_relu = F.relu(bn2b_branch2a)
        res2b_branch2b_pad = F.pad(res2b_branch2a_relu, (1, 1, 1, 1))
        res2b_branch2b  = self.res2b_branch2b(res2b_branch2b_pad)
        bn2b_branch2b   = self.bn2b_branch2b(res2b_branch2b)
        res2b_branch2b_relu = F.relu(bn2b_branch2b)
        res2b_branch2c  = self.res2b_branch2c(res2b_branch2b_relu)
        bn2b_branch2c   = self.bn2b_branch2c(res2b_branch2c)
        res2b           = res2a_relu + bn2b_branch2c
        res2b_relu      = F.relu(res2b)
        res2c_branch2a  = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a   = self.bn2c_branch2a(res2c_branch2a)
        res2c_branch2a_relu = F.relu(bn2c_branch2a)
        res2c_branch2b_pad = F.pad(res2c_branch2a_relu, (1, 1, 1, 1))
        res2c_branch2b  = self.res2c_branch2b(res2c_branch2b_pad)
        bn2c_branch2b   = self.bn2c_branch2b(res2c_branch2b)
        res2c_branch2b_relu = F.relu(bn2c_branch2b)
        res2c_branch2c  = self.res2c_branch2c(res2c_branch2b_relu)
        bn2c_branch2c   = self.bn2c_branch2c(res2c_branch2c)
        res2c           = res2b_relu + bn2c_branch2c
        res2c_relu      = F.relu(res2c)
        res3a_branch1   = self.res3a_branch1(res2c_relu)
        res3a_branch2a  = self.res3a_branch2a(res2c_relu)
        bn3a_branch1    = self.bn3a_branch1(res3a_branch1)
        bn3a_branch2a   = self.bn3a_branch2a(res3a_branch2a)
        res3a_branch2a_relu = F.relu(bn3a_branch2a)
        res3a_branch2b_pad = F.pad(res3a_branch2a_relu, (1, 1, 1, 1))
        res3a_branch2b  = self.res3a_branch2b(res3a_branch2b_pad)
        bn3a_branch2b   = self.bn3a_branch2b(res3a_branch2b)
        res3a_branch2b_relu = F.relu(bn3a_branch2b)
        res3a_branch2c  = self.res3a_branch2c(res3a_branch2b_relu)
        bn3a_branch2c   = self.bn3a_branch2c(res3a_branch2c)
        res3a           = bn3a_branch1 + bn3a_branch2c
        res3a_relu      = F.relu(res3a)
        res3b1_branch2a = self.res3b1_branch2a(res3a_relu)
        bn3b1_branch2a  = self.bn3b1_branch2a(res3b1_branch2a)
        res3b1_branch2a_relu = F.relu(bn3b1_branch2a)
        res3b1_branch2b_pad = F.pad(res3b1_branch2a_relu, (1, 1, 1, 1))
        res3b1_branch2b = self.res3b1_branch2b(res3b1_branch2b_pad)
        bn3b1_branch2b  = self.bn3b1_branch2b(res3b1_branch2b)
        res3b1_branch2b_relu = F.relu(bn3b1_branch2b)
        res3b1_branch2c = self.res3b1_branch2c(res3b1_branch2b_relu)
        bn3b1_branch2c  = self.bn3b1_branch2c(res3b1_branch2c)
        res3b1          = res3a_relu + bn3b1_branch2c
        res3b1_relu     = F.relu(res3b1)
        res3b2_branch2a = self.res3b2_branch2a(res3b1_relu)
        bn3b2_branch2a  = self.bn3b2_branch2a(res3b2_branch2a)
        res3b2_branch2a_relu = F.relu(bn3b2_branch2a)
        res3b2_branch2b_pad = F.pad(res3b2_branch2a_relu, (1, 1, 1, 1))
        res3b2_branch2b = self.res3b2_branch2b(res3b2_branch2b_pad)
        bn3b2_branch2b  = self.bn3b2_branch2b(res3b2_branch2b)
        res3b2_branch2b_relu = F.relu(bn3b2_branch2b)
        res3b2_branch2c = self.res3b2_branch2c(res3b2_branch2b_relu)
        bn3b2_branch2c  = self.bn3b2_branch2c(res3b2_branch2c)
        res3b2          = res3b1_relu + bn3b2_branch2c
        res3b2_relu     = F.relu(res3b2)
        res3b3_branch2a = self.res3b3_branch2a(res3b2_relu)
        bn3b3_branch2a  = self.bn3b3_branch2a(res3b3_branch2a)
        res3b3_branch2a_relu = F.relu(bn3b3_branch2a)
        res3b3_branch2b_pad = F.pad(res3b3_branch2a_relu, (1, 1, 1, 1))
        res3b3_branch2b = self.res3b3_branch2b(res3b3_branch2b_pad)
        bn3b3_branch2b  = self.bn3b3_branch2b(res3b3_branch2b)
        res3b3_branch2b_relu = F.relu(bn3b3_branch2b)
        res3b3_branch2c = self.res3b3_branch2c(res3b3_branch2b_relu)
        bn3b3_branch2c  = self.bn3b3_branch2c(res3b3_branch2c)
        res3b3          = res3b2_relu + bn3b3_branch2c
        res3b3_relu     = F.relu(res3b3)
        res4a_branch1   = self.res4a_branch1(res3b3_relu)
        res4a_branch2a  = self.res4a_branch2a(res3b3_relu)
        bn4a_branch1    = self.bn4a_branch1(res4a_branch1)
        bn4a_branch2a   = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a_relu = F.relu(bn4a_branch2a)
        res4a_branch2b_pad = F.pad(res4a_branch2a_relu, (1, 1, 1, 1))
        res4a_branch2b  = self.res4a_branch2b(res4a_branch2b_pad)
        bn4a_branch2b   = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b_relu = F.relu(bn4a_branch2b)
        res4a_branch2c  = self.res4a_branch2c(res4a_branch2b_relu)
        bn4a_branch2c   = self.bn4a_branch2c(res4a_branch2c)
        res4a           = bn4a_branch1 + bn4a_branch2c
        res4a_relu      = F.relu(res4a)
        res4b1_branch2a = self.res4b1_branch2a(res4a_relu)
        bn4b1_branch2a  = self.bn4b1_branch2a(res4b1_branch2a)
        res4b1_branch2a_relu = F.relu(bn4b1_branch2a)
        res4b1_branch2b_pad = F.pad(res4b1_branch2a_relu, (1, 1, 1, 1))
        res4b1_branch2b = self.res4b1_branch2b(res4b1_branch2b_pad)
        bn4b1_branch2b  = self.bn4b1_branch2b(res4b1_branch2b)
        res4b1_branch2b_relu = F.relu(bn4b1_branch2b)
        res4b1_branch2c = self.res4b1_branch2c(res4b1_branch2b_relu)
        bn4b1_branch2c  = self.bn4b1_branch2c(res4b1_branch2c)
        res4b1          = res4a_relu + bn4b1_branch2c
        res4b1_relu     = F.relu(res4b1)
        res4b2_branch2a = self.res4b2_branch2a(res4b1_relu)
        bn4b2_branch2a  = self.bn4b2_branch2a(res4b2_branch2a)
        res4b2_branch2a_relu = F.relu(bn4b2_branch2a)
        res4b2_branch2b_pad = F.pad(res4b2_branch2a_relu, (1, 1, 1, 1))
        res4b2_branch2b = self.res4b2_branch2b(res4b2_branch2b_pad)
        bn4b2_branch2b  = self.bn4b2_branch2b(res4b2_branch2b)
        res4b2_branch2b_relu = F.relu(bn4b2_branch2b)
        res4b2_branch2c = self.res4b2_branch2c(res4b2_branch2b_relu)
        bn4b2_branch2c  = self.bn4b2_branch2c(res4b2_branch2c)
        res4b2          = res4b1_relu + bn4b2_branch2c
        res4b2_relu     = F.relu(res4b2)
        res4b3_branch2a = self.res4b3_branch2a(res4b2_relu)
        bn4b3_branch2a  = self.bn4b3_branch2a(res4b3_branch2a)
        res4b3_branch2a_relu = F.relu(bn4b3_branch2a)
        res4b3_branch2b_pad = F.pad(res4b3_branch2a_relu, (1, 1, 1, 1))
        res4b3_branch2b = self.res4b3_branch2b(res4b3_branch2b_pad)
        bn4b3_branch2b  = self.bn4b3_branch2b(res4b3_branch2b)
        res4b3_branch2b_relu = F.relu(bn4b3_branch2b)
        res4b3_branch2c = self.res4b3_branch2c(res4b3_branch2b_relu)
        bn4b3_branch2c  = self.bn4b3_branch2c(res4b3_branch2c)
        res4b3          = res4b2_relu + bn4b3_branch2c
        res4b3_relu     = F.relu(res4b3)
        res4b4_branch2a = self.res4b4_branch2a(res4b3_relu)
        bn4b4_branch2a  = self.bn4b4_branch2a(res4b4_branch2a)
        res4b4_branch2a_relu = F.relu(bn4b4_branch2a)
        res4b4_branch2b_pad = F.pad(res4b4_branch2a_relu, (1, 1, 1, 1))
        res4b4_branch2b = self.res4b4_branch2b(res4b4_branch2b_pad)
        bn4b4_branch2b  = self.bn4b4_branch2b(res4b4_branch2b)
        res4b4_branch2b_relu = F.relu(bn4b4_branch2b)
        res4b4_branch2c = self.res4b4_branch2c(res4b4_branch2b_relu)
        bn4b4_branch2c  = self.bn4b4_branch2c(res4b4_branch2c)
        res4b4          = res4b3_relu + bn4b4_branch2c
        res4b4_relu     = F.relu(res4b4)
        res4b5_branch2a = self.res4b5_branch2a(res4b4_relu)
        bn4b5_branch2a  = self.bn4b5_branch2a(res4b5_branch2a)
        res4b5_branch2a_relu = F.relu(bn4b5_branch2a)
        res4b5_branch2b_pad = F.pad(res4b5_branch2a_relu, (1, 1, 1, 1))
        res4b5_branch2b = self.res4b5_branch2b(res4b5_branch2b_pad)
        bn4b5_branch2b  = self.bn4b5_branch2b(res4b5_branch2b)
        res4b5_branch2b_relu = F.relu(bn4b5_branch2b)
        res4b5_branch2c = self.res4b5_branch2c(res4b5_branch2b_relu)
        bn4b5_branch2c  = self.bn4b5_branch2c(res4b5_branch2c)
        res4b5          = res4b4_relu + bn4b5_branch2c
        res4b5_relu     = F.relu(res4b5)
        res4b6_branch2a = self.res4b6_branch2a(res4b5_relu)
        bn4b6_branch2a  = self.bn4b6_branch2a(res4b6_branch2a)
        res4b6_branch2a_relu = F.relu(bn4b6_branch2a)
        res4b6_branch2b_pad = F.pad(res4b6_branch2a_relu, (1, 1, 1, 1))
        res4b6_branch2b = self.res4b6_branch2b(res4b6_branch2b_pad)
        bn4b6_branch2b  = self.bn4b6_branch2b(res4b6_branch2b)
        res4b6_branch2b_relu = F.relu(bn4b6_branch2b)
        res4b6_branch2c = self.res4b6_branch2c(res4b6_branch2b_relu)
        bn4b6_branch2c  = self.bn4b6_branch2c(res4b6_branch2c)
        res4b6          = res4b5_relu + bn4b6_branch2c
        res4b6_relu     = F.relu(res4b6)
        res4b7_branch2a = self.res4b7_branch2a(res4b6_relu)
        bn4b7_branch2a  = self.bn4b7_branch2a(res4b7_branch2a)
        res4b7_branch2a_relu = F.relu(bn4b7_branch2a)
        res4b7_branch2b_pad = F.pad(res4b7_branch2a_relu, (1, 1, 1, 1))
        res4b7_branch2b = self.res4b7_branch2b(res4b7_branch2b_pad)
        bn4b7_branch2b  = self.bn4b7_branch2b(res4b7_branch2b)
        res4b7_branch2b_relu = F.relu(bn4b7_branch2b)
        res4b7_branch2c = self.res4b7_branch2c(res4b7_branch2b_relu)
        bn4b7_branch2c  = self.bn4b7_branch2c(res4b7_branch2c)
        res4b7          = res4b6_relu + bn4b7_branch2c
        res4b7_relu     = F.relu(res4b7)
        res4b8_branch2a = self.res4b8_branch2a(res4b7_relu)
        bn4b8_branch2a  = self.bn4b8_branch2a(res4b8_branch2a)
        res4b8_branch2a_relu = F.relu(bn4b8_branch2a)
        res4b8_branch2b_pad = F.pad(res4b8_branch2a_relu, (1, 1, 1, 1))
        res4b8_branch2b = self.res4b8_branch2b(res4b8_branch2b_pad)
        bn4b8_branch2b  = self.bn4b8_branch2b(res4b8_branch2b)
        res4b8_branch2b_relu = F.relu(bn4b8_branch2b)
        res4b8_branch2c = self.res4b8_branch2c(res4b8_branch2b_relu)
        bn4b8_branch2c  = self.bn4b8_branch2c(res4b8_branch2c)
        res4b8          = res4b7_relu + bn4b8_branch2c
        res4b8_relu     = F.relu(res4b8)
        res4b9_branch2a = self.res4b9_branch2a(res4b8_relu)
        bn4b9_branch2a  = self.bn4b9_branch2a(res4b9_branch2a)
        res4b9_branch2a_relu = F.relu(bn4b9_branch2a)
        res4b9_branch2b_pad = F.pad(res4b9_branch2a_relu, (1, 1, 1, 1))
        res4b9_branch2b = self.res4b9_branch2b(res4b9_branch2b_pad)
        bn4b9_branch2b  = self.bn4b9_branch2b(res4b9_branch2b)
        res4b9_branch2b_relu = F.relu(bn4b9_branch2b)
        res4b9_branch2c = self.res4b9_branch2c(res4b9_branch2b_relu)
        bn4b9_branch2c  = self.bn4b9_branch2c(res4b9_branch2c)
        res4b9          = res4b8_relu + bn4b9_branch2c
        res4b9_relu     = F.relu(res4b9)
        res4b10_branch2a = self.res4b10_branch2a(res4b9_relu)
        bn4b10_branch2a = self.bn4b10_branch2a(res4b10_branch2a)
        res4b10_branch2a_relu = F.relu(bn4b10_branch2a)
        res4b10_branch2b_pad = F.pad(res4b10_branch2a_relu, (1, 1, 1, 1))
        res4b10_branch2b = self.res4b10_branch2b(res4b10_branch2b_pad)
        bn4b10_branch2b = self.bn4b10_branch2b(res4b10_branch2b)
        res4b10_branch2b_relu = F.relu(bn4b10_branch2b)
        res4b10_branch2c = self.res4b10_branch2c(res4b10_branch2b_relu)
        bn4b10_branch2c = self.bn4b10_branch2c(res4b10_branch2c)
        res4b10         = res4b9_relu + bn4b10_branch2c
        res4b10_relu    = F.relu(res4b10)
        res4b11_branch2a = self.res4b11_branch2a(res4b10_relu)
        bn4b11_branch2a = self.bn4b11_branch2a(res4b11_branch2a)
        res4b11_branch2a_relu = F.relu(bn4b11_branch2a)
        res4b11_branch2b_pad = F.pad(res4b11_branch2a_relu, (1, 1, 1, 1))
        res4b11_branch2b = self.res4b11_branch2b(res4b11_branch2b_pad)
        bn4b11_branch2b = self.bn4b11_branch2b(res4b11_branch2b)
        res4b11_branch2b_relu = F.relu(bn4b11_branch2b)
        res4b11_branch2c = self.res4b11_branch2c(res4b11_branch2b_relu)
        bn4b11_branch2c = self.bn4b11_branch2c(res4b11_branch2c)
        res4b11         = res4b10_relu + bn4b11_branch2c
        res4b11_relu    = F.relu(res4b11)
        res4b12_branch2a = self.res4b12_branch2a(res4b11_relu)
        bn4b12_branch2a = self.bn4b12_branch2a(res4b12_branch2a)
        res4b12_branch2a_relu = F.relu(bn4b12_branch2a)
        res4b12_branch2b_pad = F.pad(res4b12_branch2a_relu, (1, 1, 1, 1))
        res4b12_branch2b = self.res4b12_branch2b(res4b12_branch2b_pad)
        bn4b12_branch2b = self.bn4b12_branch2b(res4b12_branch2b)
        res4b12_branch2b_relu = F.relu(bn4b12_branch2b)
        res4b12_branch2c = self.res4b12_branch2c(res4b12_branch2b_relu)
        bn4b12_branch2c = self.bn4b12_branch2c(res4b12_branch2c)
        res4b12         = res4b11_relu + bn4b12_branch2c
        res4b12_relu    = F.relu(res4b12)
        res4b13_branch2a = self.res4b13_branch2a(res4b12_relu)
        bn4b13_branch2a = self.bn4b13_branch2a(res4b13_branch2a)
        res4b13_branch2a_relu = F.relu(bn4b13_branch2a)
        res4b13_branch2b_pad = F.pad(res4b13_branch2a_relu, (1, 1, 1, 1))
        res4b13_branch2b = self.res4b13_branch2b(res4b13_branch2b_pad)
        bn4b13_branch2b = self.bn4b13_branch2b(res4b13_branch2b)
        res4b13_branch2b_relu = F.relu(bn4b13_branch2b)
        res4b13_branch2c = self.res4b13_branch2c(res4b13_branch2b_relu)
        bn4b13_branch2c = self.bn4b13_branch2c(res4b13_branch2c)
        res4b13         = res4b12_relu + bn4b13_branch2c
        res4b13_relu    = F.relu(res4b13)
        res4b14_branch2a = self.res4b14_branch2a(res4b13_relu)
        bn4b14_branch2a = self.bn4b14_branch2a(res4b14_branch2a)
        res4b14_branch2a_relu = F.relu(bn4b14_branch2a)
        res4b14_branch2b_pad = F.pad(res4b14_branch2a_relu, (1, 1, 1, 1))
        res4b14_branch2b = self.res4b14_branch2b(res4b14_branch2b_pad)
        bn4b14_branch2b = self.bn4b14_branch2b(res4b14_branch2b)
        res4b14_branch2b_relu = F.relu(bn4b14_branch2b)
        res4b14_branch2c = self.res4b14_branch2c(res4b14_branch2b_relu)
        bn4b14_branch2c = self.bn4b14_branch2c(res4b14_branch2c)
        res4b14         = res4b13_relu + bn4b14_branch2c
        res4b14_relu    = F.relu(res4b14)
        res4b15_branch2a = self.res4b15_branch2a(res4b14_relu)
        bn4b15_branch2a = self.bn4b15_branch2a(res4b15_branch2a)
        res4b15_branch2a_relu = F.relu(bn4b15_branch2a)
        res4b15_branch2b_pad = F.pad(res4b15_branch2a_relu, (1, 1, 1, 1))
        res4b15_branch2b = self.res4b15_branch2b(res4b15_branch2b_pad)
        bn4b15_branch2b = self.bn4b15_branch2b(res4b15_branch2b)
        res4b15_branch2b_relu = F.relu(bn4b15_branch2b)
        res4b15_branch2c = self.res4b15_branch2c(res4b15_branch2b_relu)
        bn4b15_branch2c = self.bn4b15_branch2c(res4b15_branch2c)
        res4b15         = res4b14_relu + bn4b15_branch2c
        res4b15_relu    = F.relu(res4b15)
        res4b16_branch2a = self.res4b16_branch2a(res4b15_relu)
        bn4b16_branch2a = self.bn4b16_branch2a(res4b16_branch2a)
        res4b16_branch2a_relu = F.relu(bn4b16_branch2a)
        res4b16_branch2b_pad = F.pad(res4b16_branch2a_relu, (1, 1, 1, 1))
        res4b16_branch2b = self.res4b16_branch2b(res4b16_branch2b_pad)
        bn4b16_branch2b = self.bn4b16_branch2b(res4b16_branch2b)
        res4b16_branch2b_relu = F.relu(bn4b16_branch2b)
        res4b16_branch2c = self.res4b16_branch2c(res4b16_branch2b_relu)
        bn4b16_branch2c = self.bn4b16_branch2c(res4b16_branch2c)
        res4b16         = res4b15_relu + bn4b16_branch2c
        res4b16_relu    = F.relu(res4b16)
        res4b17_branch2a = self.res4b17_branch2a(res4b16_relu)
        bn4b17_branch2a = self.bn4b17_branch2a(res4b17_branch2a)
        res4b17_branch2a_relu = F.relu(bn4b17_branch2a)
        res4b17_branch2b_pad = F.pad(res4b17_branch2a_relu, (1, 1, 1, 1))
        res4b17_branch2b = self.res4b17_branch2b(res4b17_branch2b_pad)
        bn4b17_branch2b = self.bn4b17_branch2b(res4b17_branch2b)
        res4b17_branch2b_relu = F.relu(bn4b17_branch2b)
        res4b17_branch2c = self.res4b17_branch2c(res4b17_branch2b_relu)
        bn4b17_branch2c = self.bn4b17_branch2c(res4b17_branch2c)
        res4b17         = res4b16_relu + bn4b17_branch2c
        res4b17_relu    = F.relu(res4b17)
        res4b18_branch2a = self.res4b18_branch2a(res4b17_relu)
        bn4b18_branch2a = self.bn4b18_branch2a(res4b18_branch2a)
        res4b18_branch2a_relu = F.relu(bn4b18_branch2a)
        res4b18_branch2b_pad = F.pad(res4b18_branch2a_relu, (1, 1, 1, 1))
        res4b18_branch2b = self.res4b18_branch2b(res4b18_branch2b_pad)
        bn4b18_branch2b = self.bn4b18_branch2b(res4b18_branch2b)
        res4b18_branch2b_relu = F.relu(bn4b18_branch2b)
        res4b18_branch2c = self.res4b18_branch2c(res4b18_branch2b_relu)
        bn4b18_branch2c = self.bn4b18_branch2c(res4b18_branch2c)
        res4b18         = res4b17_relu + bn4b18_branch2c
        res4b18_relu    = F.relu(res4b18)
        res4b19_branch2a = self.res4b19_branch2a(res4b18_relu)
        bn4b19_branch2a = self.bn4b19_branch2a(res4b19_branch2a)
        res4b19_branch2a_relu = F.relu(bn4b19_branch2a)
        res4b19_branch2b_pad = F.pad(res4b19_branch2a_relu, (1, 1, 1, 1))
        res4b19_branch2b = self.res4b19_branch2b(res4b19_branch2b_pad)
        bn4b19_branch2b = self.bn4b19_branch2b(res4b19_branch2b)
        res4b19_branch2b_relu = F.relu(bn4b19_branch2b)
        res4b19_branch2c = self.res4b19_branch2c(res4b19_branch2b_relu)
        bn4b19_branch2c = self.bn4b19_branch2c(res4b19_branch2c)
        res4b19         = res4b18_relu + bn4b19_branch2c
        res4b19_relu    = F.relu(res4b19)
        res4b20_branch2a = self.res4b20_branch2a(res4b19_relu)
        bn4b20_branch2a = self.bn4b20_branch2a(res4b20_branch2a)
        res4b20_branch2a_relu = F.relu(bn4b20_branch2a)
        res4b20_branch2b_pad = F.pad(res4b20_branch2a_relu, (1, 1, 1, 1))
        res4b20_branch2b = self.res4b20_branch2b(res4b20_branch2b_pad)
        bn4b20_branch2b = self.bn4b20_branch2b(res4b20_branch2b)
        res4b20_branch2b_relu = F.relu(bn4b20_branch2b)
        res4b20_branch2c = self.res4b20_branch2c(res4b20_branch2b_relu)
        bn4b20_branch2c = self.bn4b20_branch2c(res4b20_branch2c)
        res4b20         = res4b19_relu + bn4b20_branch2c
        res4b20_relu    = F.relu(res4b20)
        res4b21_branch2a = self.res4b21_branch2a(res4b20_relu)
        bn4b21_branch2a = self.bn4b21_branch2a(res4b21_branch2a)
        res4b21_branch2a_relu = F.relu(bn4b21_branch2a)
        res4b21_branch2b_pad = F.pad(res4b21_branch2a_relu, (1, 1, 1, 1))
        res4b21_branch2b = self.res4b21_branch2b(res4b21_branch2b_pad)
        bn4b21_branch2b = self.bn4b21_branch2b(res4b21_branch2b)
        res4b21_branch2b_relu = F.relu(bn4b21_branch2b)
        res4b21_branch2c = self.res4b21_branch2c(res4b21_branch2b_relu)
        bn4b21_branch2c = self.bn4b21_branch2c(res4b21_branch2c)
        res4b21         = res4b20_relu + bn4b21_branch2c
        res4b21_relu    = F.relu(res4b21)
        res4b22_branch2a = self.res4b22_branch2a(res4b21_relu)
        bn4b22_branch2a = self.bn4b22_branch2a(res4b22_branch2a)
        res4b22_branch2a_relu = F.relu(bn4b22_branch2a)
        res4b22_branch2b_pad = F.pad(res4b22_branch2a_relu, (1, 1, 1, 1))
        res4b22_branch2b = self.res4b22_branch2b(res4b22_branch2b_pad)
        bn4b22_branch2b = self.bn4b22_branch2b(res4b22_branch2b)
        res4b22_branch2b_relu = F.relu(bn4b22_branch2b)
        res4b22_branch2c = self.res4b22_branch2c(res4b22_branch2b_relu)
        bn4b22_branch2c = self.bn4b22_branch2c(res4b22_branch2c)
        res4b22         = res4b21_relu + bn4b22_branch2c
        res4b22_relu    = F.relu(res4b22)
        return res4b22_relu


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        #if 'scale' in __bottom_layers_weights_dict[name]:
        #    layer.state_dict()['weight'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['scale']))
        #else:
        #    layer.weight.data.fill_(1)

        #if 'bias' in __bottom_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['bias']))
        #else:
        #    layer.bias.data.fill_(0)

        #layer.state_dict()['running_mean'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['mean']))
        #layer.state_dict()['running_var'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        #layer.state_dict()['weight'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['weights']))
        #if 'bias' in __bottom_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__bottom_layers_weights_dict[name]['bias']))
        return layer





class TopLayers(nn.Module):
    def __init__(self, weight_file=None):
        super(TopLayers, self).__init__()
        #global __top_layers_weights_dict  # Loaded by MMDNN
        #__top_layers_weights_dict = load_weights(weight_file)

        self.res5a_branch1 = self.__conv(2, name='res5a_branch1', in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res5a_branch2a = self.__conv(2, name='res5a_branch2a', in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn5a_branch1 = self.__batch_normalization(2, 'bn5a_branch1', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.bn5a_branch2a = self.__batch_normalization(2, 'bn5a_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5a_branch2b = self.__conv(2, name='res5a_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2b = self.__batch_normalization(2, 'bn5a_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5a_branch2c = self.__conv(2, name='res5a_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2c = self.__batch_normalization(2, 'bn5a_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2a = self.__conv(2, name='res5b_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2a = self.__batch_normalization(2, 'bn5b_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2b = self.__conv(2, name='res5b_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2b = self.__batch_normalization(2, 'bn5b_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2c = self.__conv(2, name='res5b_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2c = self.__batch_normalization(2, 'bn5b_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2a = self.__conv(2, name='res5c_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2a = self.__batch_normalization(2, 'bn5c_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2b = self.__conv(2, name='res5c_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2b = self.__batch_normalization(2, 'bn5c_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2c = self.__conv(2, name='res5c_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2c = self.__batch_normalization(2, 'bn5c_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.bbox_pred_1 = self.__dense(name = 'bbox_pred_1', in_features = 2048, out_features = 8, bias = True)
        self.cls_score_1 = self.__dense(name = 'cls_score_1', in_features = 2048, out_features = 2, bias = True)

    def forward(self, x):
        res5a_branch1   = self.res5a_branch1(x)
        res5a_branch2a  = self.res5a_branch2a(x)
        bn5a_branch1    = self.bn5a_branch1(res5a_branch1)
        bn5a_branch2a   = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a_relu = F.relu(bn5a_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5a_branch2b_pad = F.pad(res5a_branch2a_relu, (0, 0, 0, 0))
        #res5a_branch2b  = self.res5a_branch2b(res5a_branch2b_pad)
        # Fix broken dilated convolutions on MMDNN conversion, and roll in padding
        res5a_branch2b = F.conv2d(res5a_branch2a_relu, weight=self.res5a_branch2b.weight, bias=self.res5a_branch2b.bias, 
                                  stride=self.res5a_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5a_branch2b.groups)

        bn5a_branch2b   = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b_relu = F.relu(bn5a_branch2b)
        res5a_branch2c  = self.res5a_branch2c(res5a_branch2b_relu)
        bn5a_branch2c   = self.bn5a_branch2c(res5a_branch2c)
        res5a           = bn5a_branch1 + bn5a_branch2c
        res5a_relu      = F.relu(res5a)
        res5b_branch2a  = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a   = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a_relu = F.relu(bn5b_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5b_branch2b_pad = F.pad(res5b_branch2a_relu, (0, 0, 0, 0))
        #res5b_branch2b  = self.res5b_branch2b(res5b_branch2b_pad)
        res5b_branch2b = F.conv2d(res5b_branch2a_relu, weight=self.res5b_branch2b.weight, bias=self.res5b_branch2b.bias, 
                                  stride=self.res5b_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5b_branch2b.groups)

        bn5b_branch2b   = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b_relu = F.relu(bn5b_branch2b)
        res5b_branch2c  = self.res5b_branch2c(res5b_branch2b_relu)
        bn5b_branch2c   = self.bn5b_branch2c(res5b_branch2c)
        res5b           = res5a_relu + bn5b_branch2c
        res5b_relu      = F.relu(res5b)
        res5c_branch2a  = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a   = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a_relu = F.relu(bn5c_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5c_branch2b_pad = F.pad(res5c_branch2a_relu, (1, 1, 1, 1))
        #res5c_branch2b  = self.res5c_branch2b(res5c_branch2b_pad)
        res5c_branch2b = F.conv2d(res5c_branch2a_relu, weight=self.res5c_branch2b.weight, bias=self.res5c_branch2b.bias, 
                                  stride=self.res5c_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5c_branch2b.groups)
        bn5c_branch2b   = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b_relu = F.relu(bn5c_branch2b)
        res5c_branch2c  = self.res5c_branch2c(res5c_branch2b_relu)
        bn5c_branch2c   = self.bn5c_branch2c(res5c_branch2c)
        res5c           = res5b_relu + bn5c_branch2c
        res5c_relu      = F.relu(res5c)
        pool5           = F.avg_pool2d(res5c_relu, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        bbox_pred_0     = pool5.view(pool5.size(0), -1)
        cls_score_0     = pool5.view(pool5.size(0), -1)
        bbox_pred_1     = self.bbox_pred_1(bbox_pred_0)
        cls_score_1     = self.cls_score_1(cls_score_0)
        # import pdb; pdb.set_trace()
        cls_prob        = F.softmax(cls_score_1, dim=1)
        # Returning pre-softmax score to be consistent with Caffe implementation
        return bbox_pred_1, cls_prob, cls_score_1

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        #if 'scale' in __top_layers_weights_dict[name]:
        #    layer.state_dict()['weight'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['scale']))
        #else:
        #    layer.weight.data.fill_(1)

        #if 'bias' in __top_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['bias']))
        #else:
        #    layer.bias.data.fill_(0)

        #layer.state_dict()['running_mean'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['mean']))
        #layer.state_dict()['running_var'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:  
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:  
            layer = nn.Conv3d(**kwargs)
        else:           
            raise NotImplementedError()

        #layer.state_dict()['weight'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['weights']))
        #if 'bias' in __top_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        #layer.state_dict()['weight'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['weights']))
        #if 'bias' in __top_layers_weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__top_layers_weights_dict[name]['bias']))
        return layer



class FasterRCNN(nn.Module):
    """PyTorch-1.3 model conversion of ResNet-101_faster_rcnn_ohem_iter_20000.caffemodel, leveraging MMDNN conversion tools"""
    def __init__(self, device):
        super(FasterRCNN, self).__init__()

        self.device = device
        torch.device(device)

        self.top = TopLayers()
        self.top.eval()
        self.top = self.top.to(device)

        self.bottom = BottomLayers()
        self.bottom.eval()
        self.bottom = self.bottom.to(device)

        self.rpn = RpnLayers()
        self.rpn.eval()
        self.rpn = self.rpn.to(device)

        # Proposal layer:  manually imported from janus/src/rpn
        self._feat_stride = 16
        #self._anchors = rpn.generate_anchors.generate_anchors(scales=np.array( (8,16,32) ))    
        self._anchors = np.array([[ -84.,  -40.,   99.,   55.],
                                  [-176.,  -88.,  191.,  103.],
                                  [-360., -184.,  375.,  199.],
                                  [ -56.,  -56.,   71.,   71.],
                                  [-120., -120.,  135.,  135.],
                                  [-248., -248.,  263.,  263.],
                                  [ -36.,  -80.,   51.,   95.],
                                  [ -80., -168.,   95.,  183.],
                                  [-168., -344.,  183.,  359.]])
        self._num_anchors = self._anchors.shape[0]
        
    def __call__(self, im, im_info):
        # im is a tensor, N x 3 x H x W; im_info is another,
        # N x 3 (H, W, scale)
        with torch.no_grad():
            im = im.to(self.device)
            
            res4b22 = self.bottom(im)
            
            (rpn_cls_score, rpn_bbox_pred) = self.rpn(res4b22)
            
            (N,C,W,H) = rpn_cls_score.shape
            rpn_cls_score_reshape = torch.reshape(rpn_cls_score, (N, 2, -1, H))
            del rpn_cls_score
            rpn_cls_prob = torch.nn.functional.softmax(rpn_cls_score_reshape, dim=1)  # FIXME: is this dim right?
            rpn_cls_prob_reshape = torch.reshape(rpn_cls_prob, (N, 18, -1, H))
            del rpn_cls_prob
            
            # TODO: Make this handle multiple images, instead of horrible flaming death.
            rois = self._proposal_layer(rpn_cls_prob_reshape.cpu(), rpn_bbox_pred.cpu(), im_info)
            del rpn_bbox_pred
            # import pdb; pdb.set_trace()
            rois_gpu = rois.to(self.device)
            roi_pool5 = torchvision.ops.roi_pool(res4b22, rois_gpu, (14,14), 0.0625)
            del res4b22
            del rois_gpu
            (bbox_pred_1, cls_prob, cls_score) = self.top(roi_pool5)
            rois_cpu = rois.cpu()
            del rois
            bbox_pred_1_cpu = bbox_pred_1.cpu()
            del bbox_pred_1
            cls_prob_cpu = cls_prob.cpu()
            del cls_prob
            cls_score_cpu = cls_score.cpu()
            del cls_score
            del im
        return (rois_cpu, bbox_pred_1_cpu, cls_prob_cpu, cls_score_cpu)

    def _proposal_layer(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """rpn.proposal_layer"""
        # 'Only single item batches are supported'
        assert(rpn_cls_prob_reshape.shape[0] == 1) 

        # TODO: Defaults from caffe: make me configurable 
        #   {'PROPOSAL_METHOD': 'selective_search', 'SVM': False, 'NMS': 0.3, 'RPN_NMS_THRESH': 0.7, 'SCALES': [800], 
        #   'RPN_POST_NMS_TOP_N': 300, 'HAS_RPN': False, 'RPN_PRE_NMS_TOP_N': 6000, 'BBOX_REG': True, 'RPN_MIN_SIZE': 3, 'MAX_SIZE': 1333}
        cfg_key = 'TEST'      # either 'TRAIN' or 'TEST'
        pre_nms_topN = 6000   # RPN_PRE_NMS_TOP_N
        post_nms_topN = 300   # RPN_POST_NMS_TOP_N
        nms_thresh= 0.7       # RPN_NMS_THRESH
        min_size = 3          # RPN_MIN_SIZE                

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = rpn_cls_prob_reshape.detach().numpy()[:, self._num_anchors:, :, :]
        bbox_deltas = rpn_bbox_pred.detach().numpy()
        (im_height, im_width, im_scale) = im_info[0]  # H, W, scale

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = self._bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = self._clip_boxes(proposals, (im_height.item(), im_width.item()))

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self._filter_boxes(proposals, min_size * im_scale.item())
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = self._nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return torch.tensor(blob)


    def _bbox_transform_inv(self, boxes, deltas):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.bbox_transform_inv"""
        if boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = boxes.astype(deltas.dtype, copy=False)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes

    def _clip_boxes(self, boxes, im_shape):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.clip_boxes"""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes


    def _filter_boxes(self, boxes, min_size):
        """Cloned from janus-tne/src/python/rpn.proposal_layer._filter_boxes"""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep


    def _nms(self, dets, thresh):
        """Cloned from janus-tne/src/python/nms/py_cpu_nms.py"""
        """FIXME: GPU acceleration needed?"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        
        return keep




class FasterRCNN_MMDNN(nn.Module):
    """PyTorch-1.3 model conversion of ResNet-101_faster_rcnn_ohem_iter_20000.caffemodel, leveraging MMDNN conversion tools"""
    def __init__(self, model_dir, device):
        super(FasterRCNN_MMDNN, self).__init__()

        self.device = device
        torch.device(device)
        # Converted using convert_caffe_to_pytorch.convert_top()
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'detection')
        MainModel = imp.load_source('MainModel', os.path.join(dir, "top_layers.py"))
        self.top = torch.load(os.path.join(model_dir, "top_layers.pth"), map_location=device)
        self.top.eval()

        # Converted using convert_caffe_to_pytorch.convert_bottom()
        MainModel = imp.load_source('MainModel', os.path.join(dir, "bottom_layers.py"))
        self.bottom = torch.load(os.path.join(model_dir, "bottom_layers.pth"), map_location=device)
        self.bottom.eval()
        self.bottom = self.bottom.to(device)

        # Converted using convert_caffe_to_pytorch.convert_rpn()
        MainModel = imp.load_source('MainModel', os.path.join(dir, "rpn_layers.py"))
        self.rpn = torch.load(os.path.join(model_dir, "rpn_layers.pth"), map_location=device)
        self.rpn.eval()
        self.rpn = self.rpn.to(device)


        # Proposal layer:  manually imported from janus/src/rpn
        self._feat_stride = 16
        #self._anchors = rpn.generate_anchors.generate_anchors(scales=np.array( (8,16,32) ))    
        self._anchors = np.array([[ -84.,  -40.,   99.,   55.],
                                  [-176.,  -88.,  191.,  103.],
                                  [-360., -184.,  375.,  199.],
                                  [ -56.,  -56.,   71.,   71.],
                                  [-120., -120.,  135.,  135.],
                                  [-248., -248.,  263.,  263.],
                                  [ -36.,  -80.,   51.,   95.],
                                  [ -80., -168.,   95.,  183.],
                                  [-168., -344.,  183.,  359.]])
        self._num_anchors = self._anchors.shape[0]
        
    def __call__(self, im, im_info):
        # im is a tensor, N x 3 x H x W; im_info is another,
        # N x 3 (H, W, scale)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            im = im.to(self.device)
            
            res4b22 = self.bottom(im)
            
            (rpn_cls_score, rpn_bbox_pred) = self.rpn(res4b22)
            
            (N,C,W,H) = rpn_cls_score.shape
            rpn_cls_score_reshape = torch.reshape(rpn_cls_score, (N, 2, -1, H))
            del rpn_cls_score
            rpn_cls_prob = torch.nn.functional.softmax(rpn_cls_score_reshape, dim=1)  # FIXME: is this dim right?
            rpn_cls_prob_reshape = torch.reshape(rpn_cls_prob, (N, 18, -1, H))
            del rpn_cls_prob
            
            # TODO: Make this handle multiple images, instead of horrible flaming death.
            rois = self._proposal_layer(rpn_cls_prob_reshape.cpu(), rpn_bbox_pred.cpu(), im_info)
            del rpn_bbox_pred
            # import pdb; pdb.set_trace()
            rois_gpu = rois.to(self.device)
            roi_pool5 = torchvision.ops.roi_pool(res4b22, rois_gpu, (14,14), 0.0625)
            del res4b22
            del rois_gpu
            (bbox_pred_1, cls_prob, cls_score) = self.top(roi_pool5)
            rois_cpu = rois.cpu()
            del rois
            bbox_pred_1_cpu = bbox_pred_1.cpu()
            del bbox_pred_1
            cls_prob_cpu = cls_prob.cpu()
            del cls_prob
            cls_score_cpu = cls_score.cpu()   
            del cls_score
            del im
        return (rois_cpu, bbox_pred_1_cpu, cls_prob_cpu, cls_score_cpu)

    def _proposal_layer(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """rpn.proposal_layer"""
        # 'Only single item batches are supported'
        assert(rpn_cls_prob_reshape.shape[0] == 1) 

        # TODO: Defaults from caffe: make me configurable 
        #   {'PROPOSAL_METHOD': 'selective_search', 'SVM': False, 'NMS': 0.3, 'RPN_NMS_THRESH': 0.7, 'SCALES': [800], 
        #   'RPN_POST_NMS_TOP_N': 300, 'HAS_RPN': False, 'RPN_PRE_NMS_TOP_N': 6000, 'BBOX_REG': True, 'RPN_MIN_SIZE': 3, 'MAX_SIZE': 1333}
        cfg_key = 'TEST'      # either 'TRAIN' or 'TEST'
        pre_nms_topN = 6000   # RPN_PRE_NMS_TOP_N
        post_nms_topN = 300   # RPN_POST_NMS_TOP_N
        nms_thresh= 0.7       # RPN_NMS_THRESH
        min_size = 3          # RPN_MIN_SIZE                

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = rpn_cls_prob_reshape.detach().numpy()[:, self._num_anchors:, :, :]
        bbox_deltas = rpn_bbox_pred.detach().numpy()
        (im_height, im_width, im_scale) = im_info[0]  # H, W, scale

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = self._bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = self._clip_boxes(proposals, (im_height.item(), im_width.item()))

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self._filter_boxes(proposals, min_size * im_scale.item())
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = self._nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return torch.tensor(blob)


    def _bbox_transform_inv(self, boxes, deltas):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.bbox_transform_inv"""
        if boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = boxes.astype(deltas.dtype, copy=False)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes

    def _clip_boxes(self, boxes, im_shape):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.clip_boxes"""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes


    def _filter_boxes(self, boxes, min_size):
        """Cloned from janus-tne/src/python/rpn.proposal_layer._filter_boxes"""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep


    def _nms(self, dets, thresh):
        """Cloned from janus-tne/src/python/nms/py_cpu_nms.py"""
        """FIXME: GPU acceleration needed?"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        
        return keep


def conversion():
    mmdnn = FasterRCNN_MMDNN('./models/detection', 'cpu'); 
    net = FasterRCNN('cpu')
    net.load_state_dict(mmdnn.state_dict())
    torch.save(net.state_dict(), './models/detection/faster_rcnn.pth')
    return net
