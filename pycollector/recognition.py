import os
import sys
import torch
import vipy
import shutil
import numpy as np
from vipy.util import remkdir, filetail, readlist, tolist, filepath
from pycollector.video import Video
from pycollector.model.yolov3.network import Darknet
from pycollector.globals import print
from pycollector.model.pyvideoresearch.bases.resnet50_3d import ResNet503D, ResNet3D, Bottleneck3D


class ActivityRecognition(object):
    def __init__(self):
        #weights = 'visym/models/train_resnet50_3d_pretrain_pip50k_finetune_betterbox_meva/model_best.pth.tar' 
        weights = 'resnet50_3d_pretrain_pip50k_finetune_betterbox_meva.pth.tar'
        os.system('wget -c https://dl.dropboxusercontent.com/s/a0ouihgjxiwn2k8/resnet50_3d_pretrain_pip50k_finetune_betterbox_meva.pth.tar -O %s' % weights) 
        d = {k.replace('module.basenet.0.',''):v for (k,v) in torch.load(weights)['state_dict'].items()}
        self.net =  ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes=53).load_state_dict(d)
