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
import pycollector.model.ResNets_3D_PyTorch.resnet
import pycollector.label

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl




class ActivityRecognition(object):
    def __init__(self, pretrained=True):
        self.net =  None
        self._class_to_index = {}
        self._num_frames = 0

    def class_to_index(self, c=None):
        return self._class_to_index if c is None else self._class_to_index[c]
    
    def index_to_class(self, index=None):
        d = {v:k for (k,v) in self.class_to_index().items()}
        return d if index is None else d[index]
    
    def __call__(self, video=None, tensor=None):
        assert self.net is not None, "Invalid network"
        assert video is not None or tensor is not None
        return self.net(tensor) if tensor is not None else self.net(self.totensor(video))

    def forward(self, x):
        return self.__call__(tensor=x)
    
    def classlist(self):
        return [k for (k,v) in sorted(list(self.class_to_index().items()), key=lambda x: x[0])]  # sorted in index order

    def num_classes(self):
        return len(self.classlist())

    def fromindex(self, k):
        index_to_class = {v:k for (k,v) in self.class_to_index().items()}
        assert k in index_to_class, "Invalid class index '%s'" % (str(k))
        return index_to_class[k]

    def label_confidence(self, video=None, tensor=None, threshold=None):
        logits = self.__call__(video, tensor)
        conf = [[(self.index_to_class(j), s[j]) for j in i[::-1] if threshold is None or s[j]>threshold] for (s,i) in zip(logits, np.argsort(logits, axis=1))]
        return conf if len(logits) > 1 else conf[0]

    def activity(self, video, threshold=None):
        (c,s) = zip(*self.label_confidence(video=video, threshold=None))
        return vipy.activity.Activity(startframe=0, endframe=self._num_frames, category=c[0], actorid=video.actorid(), confidence=s[0]) if (threshold is None or s[0]>threshold) else None
            
    def top1(self, video=None, tensor=None, threshold=None):
        return self.topk(k=1, video=video, tensor=tensor, threshold=threshold)

    def topk(self, k, video=None, tensor=None, threshold=None):
        logits = self.__call__(video, tensor)
        topk = [[self.index_to_class(j) for j in i[-k:][::-1] if threshold is None or s[j] >= threshold] for (s,i) in zip(logits, np.argsort(logits, axis=1))]
        return topk if len(topk) > 1 else topk[0]

    def temporal_support(self):
        return self._num_frames

    def totensor(self, training=False):
        raise

    def binary_vector(self, categories):
        y = np.zeros(len(self.classlist())).astype(np.float32)
        for c in tolist(categories):
            y[self.class_to_index(c)] = 1
        return torch.from_numpy(y).type(torch.FloatTensor)
        
    
class MevaActivityRecognition(ActivityRecognition):
    def __init__(self, weightfile=None):
        indir = remkdir(os.path.join(filepath(os.path.abspath(__file__)), 'model', 'recognition'))
        
        #weightfile = os.path.join(indir, 'resnet50_3d_pretrain_pip50k_finetune_betterbox_meva.pth.tar') if weightfile is None else weightfile
        #if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, 'a3ecabe5e95e6d603c0c47e68bdd524e82503a88'):
        #    os.system('wget -c https://dl.dropboxusercontent.com/s/a0ouihgjxiwn2k8/resnet50_3d_pretrain_pip50k_finetune_betterbox_meva.pth.tar -O %s' % weightfile)
        #d = {k.replace('module.basenet.0.',''):v for (k,v) in torch.load(weightfile, map_location=torch.device('cpu'))['state_dict'].items()}
        
        weightfile = os.path.join(indir, 'resnet50_3d_pretrain_pip50k_finetune_betterbox.pth.tar') if weightfile is None else weightfile
        if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, '5103f4baa617603f8f6ea08a21e9c7f41f793b69'):
            os.system('wget -c https://dl.dropboxusercontent.com/s/gbp992h1vob6duy/resnet50_3d_pretrain_pip50k_finetune_betterbox.pth.tar -O %s' % weightfile) 
        d = {k.replace('module.basenet.',''):v for (k,v) in torch.load(weightfile, map_location=torch.device('cpu'))['state_dict'].items()}    

        self.net =  ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes=53)
        self.net.load_state_dict(d)
        self.net.eval()

        self._input_size = 224
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._num_frames = 64

        self._class_to_index =  {'car_drops_off_person': 0,
                                 'car_makes_u_turn': 1,
                                 'car_picks_up_person': 2,
                                 'car_reverses': 3,
                                 'car_starts': 4,
                                 'car_stops': 5,
                                 'car_turns_left': 6,
                                 'car_turns_right': 7,
                                 'hand_interacts_with_person_highfive': 8,
                                 'hand_interacts_with_person_holdhands': 9,
                                 'hand_interacts_with_person_shakehands': 10,
                                 'motorcycle_drops_off_person': 11,
                                 'motorcycle_makes_u_turn': 12,
                                 'motorcycle_picks_up_person': 13,
                                 'motorcycle_reverses': 14,
                                 'motorcycle_starts': 15,
                                 'motorcycle_stops': 16,
                                 'motorcycle_turns_left': 17,
                                 'motorcycle_turns_right': 18,
                                 'person_abandons_package': 19,
                                 'person_carries_heavy_object': 20,
                                 'person_closes_car_door': 21,
                                 'person_closes_car_trunk': 22,
                                 'person_closes_facility_door': 23,
                                 'person_closes_motorcycle_trunk': 24,
                                 'person_embraces_person': 25,
                                 'person_enters_car': 26,
                                 'person_enters_scene_through_structure': 27,
                                 'person_exits_car': 28,
                                 'person_exits_scene_through_structure': 29,
                                 'person_interacts_with_laptop': 30,
                                 'person_loads_car': 31,
                                 'person_loads_motorcycle': 32,
                                 'person_opens_car_door': 33,
                                 'person_opens_car_trunk': 34,
                                 'person_opens_facility_door': 35,
                                 'person_opens_motorcycle_trunk': 36,
                                 'person_picks_up_object': 37,
                                 'person_purchases_from_machine': 38,
                                 'person_puts_down_object': 39,
                                 'person_reads_document': 40,
                                 'person_rides_bicycle': 41,
                                 'person_sits_down': 42,
                                 'person_stands_up': 43,
                                 'person_steals_object': 44,
                                 'person_talks_on_phone': 45,
                                 'person_talks_to_person': 46,
                                 'person_texts_on_phone': 47,
                                 'person_transfers_object_to_car': 48,
                                 'person_transfers_object_to_motorcycle': 49,
                                 'person_transfers_object_to_person': 50,
                                 'person_unloads_car': 51,
                                 'person_unloads_motorcycle': 52}
        
        
        
    def __call__(self, video=None, tensor=None):
        assert video is not None or tensor is not None, "invalid input - must provide at least video or tensor input"
        assert not (video is not None and tensor is not None), "Invalid input - must provide at most one input"
        assert len(tolist(video)) == 1, "Single element batch only for now..."
        
        if video is not None:
            assert video.width() == video.height(), "Video must be square"
            v = video.resize(self._input_size, self._input_size).centercrop( (self._input_size, self._input_size) )
            v = v.normalize(mean=self._mean, std=self._std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load()                                                                                                                                                   
            tensor = v.torch(startframe=0, length=self._num_frames, boundary='repeat', order='nhwc')
            tensor = tensor.unsqueeze(0)

        return self.net.forward(tensor).mean(1).tolist()  # global temporal pooling


    def tomeva(self, cls):
        d_pip_to_meva = pycollector.label.pip_to_meva()
        return d_pip_to_meva[cls]

    
class PIP_250k(pl.LightningModule, ActivityRecognition):
    """Activity recognition using people in public - 250k"""
    
    def __init__(self, pretrained=True, deterministic=False):
        super().__init__()
        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

        if deterministic:
            np.random.seed(42)

        # Powerset
        self._class_to_index = pycollector.label.pip_250k_powerset()
        
        if pretrained:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())

    def category(self, x):
        yh = self.forward(x)
        return [self.index_to_class(int(k)) for (c,k) in torch.max(yh, dim=1)]

    def category_confidence(self, x):
        yh = self.forward(x)
        return [(self.index_to_class(int(k)), float(c)) for (c,k) in torch.max(yh, dim=1)]

    # ---- <LIGHTNING>
    def forward(self, x):
        return self.net(x)  # lighting handles device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_nb):
        (x,y) = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        (x,y) = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
    # ---- </LIGHTNING>
    
    @classmethod
    def from_checkpoint(cls, checkpointpath):
        return cls().load_from_checkpoint(checkpointpath)  # lightning
            
    def _load_trained(self):
        raise
        
    def _load_pretrained(self):

        pthfile = vipy.util.tocache('r3d50_KMS_200ep.pth')
        if not os.path.exists(pthfile) or not vipy.downloader.verify_sha1(pthfile, '39ea626355308d8f75307cab047a8d75862c3261'):
            print('[pycollector.recognition]: Downloading pretrained weights ...')
            os.system('wget -c https://dl.dropboxusercontent.com/s/t3xge6lrfqpklr0/r3d50_kms_200ep.pth -O %s' % pthfile) 
        assert vipy.downloader.verify_sha1(pthfile, '39ea626355308d8f75307cab047a8d75862c3261'), "SHA1 check failed"

        net = pycollector.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=1139)
        pretrain = torch.load(pthfile, map_location='cpu')
        net.load_state_dict(pretrain['state_dict'])

        # Inflate RGB -> RGBA         
        t = torch.split(net.conv1.weight.data, dim=1, split_size_or_sections=1)
        net.conv1.weight.data = torch.cat( (*t, t[-1]), dim=1).contiguous()
        net.conv1.in_channels = 4

        self.net = net

        return self

    @staticmethod
    def _totensor(v, training, validation, input_size, num_frames, mean, std, d_class_to_index=None):
        #assert isinstance(v, vipy.video.Scene), "Invalid input"

        max_frames = v.duration_in_frames_of_videofile()
        startframe = np.random.randint(max_frames-num_frames-1) if (max_frames-num_frames-1)>0 else 0
        v = v.clip(startframe, startframe+num_frames)
        v = v.crop(v.trackbox().maxsquare())

        if training:
            v = v.fliplr() if np.random.rand() > 0.5 else v
            v = v.resize(input_size, input_size)  
        elif validation:
            v = v.resize(input_size, input_size)
        else:
            v = v.resize(input_size, input_size)

        v = v.load().normalize(mean=mean, std=std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load()        
        (t,lbl) = v.torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw', withlabel=True)  # (c=3)x(d=num_frames)x(H=input_size)x(W=input_size) 
        b = v.clone().binarymask().bias(-0.5).torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw')  # (c=1)x(d=num_frames)x(H=input_size)x(W=input_size), in [-0.5, 0.5]
        t = torch.cat((t,b), dim=0)  # (c=4) x (d=num_frames) x (H=input_size) x (W=input_size)

        if training or validation:
            assert d_class_to_index is not None, "Invalid input"
            return (t, d_class_to_index[vipy.util.most_frequent(lbl)])
        else:
            return t

    def totensor(self, v=None, training=False, validation=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, d=self._class_to_index: PIP_250k._totensor(v, training, validation, input_size, num_frames, mean, std, d)
        return f(v) if v is not None else f
    
    def train(self, trainset, valset=None, gpus=None, num_workers=1, batch_size=1, outdir='.'):
        """Default training options"""
        assert isinstance(trainset, pycollector.dataset.Dataset)
        assert set(self.classlist()) == trainset.classlist()
        assert valset is None or isinstance(valset, pycollector.dataset.Dataset)                
        assert valset is None or set(valset.classlist()).issubset(set(self.classlist()))
        
        trainloader = DataLoader(trainset.totorch(self.totensor(training=True), num_workers=num_workers, batch_size=batch_size, pin_memory=True))
        valloader = None if valset is None else DataLoader(valset.totorch(self.totensor(validation=True), num_workers=num_workers, batch_size=batch_size, pin_memory=True))
        return pl.Trainer(gpus=gpus, accelerator='dp', default_root_dir=outdir).fit(self, trainloader, valloader)
