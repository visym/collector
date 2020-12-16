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
import pycollector.dataset

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl
import json



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

        self._class_to_weight = {'car_drops_off_person': 0.5415580813018052, 'car_picks_up_person': 0.5166285035486872, 'car_reverses': 0.4864531915292076, 'car_starts': 0.4373510029809646, 'car_stops': 0.2710265819171847, 'car_turns_left': 0.8712633574636675, 'car_turns_right': 0.3827979879057864, 'hand_interacts_with_person_highfive': 1.4962896036348539, 'person': 0.08372274235167537, 'person_abandons_object': 0.5294114478840813, 'person_carries_heavy_object': 0.3982262588212442, 'person_closes_car_door': 0.30026000397668684, 'person_closes_car_trunk': 0.6329073059937789, 'person_closes_facility_door': 0.3572828721142877, 'person_embraces_person': 0.321974093136734, 'person_enters_car': 0.30481062242606516, 'person_enters_scene_through_structure': 0.31850059486555565, 'person_exits_car': 0.4071776409464214, 'person_exits_scene_through_structure': 0.38887859050533025, 'person_holds_hand': 0.37311850556431675, 'person_interacts_with_laptop': 0.5288033409750182, 'person_loads_car': 1.217126295244549, 'person_opens_car_door': 0.24504116978076299, 'person_opens_car_trunk': 1.0934851888105013, 'person_opens_facility_door': 0.4955138087004537, 'person_picks_up_object_from_floor': 1.1865967553059953, 'person_picks_up_object_from_table': 3.9581702614257934, 'person_purchases_from_cashier': 3.795407604809639, 'person_purchases_from_machine': 3.013281383565209, 'person_puts_down_object_on_floor': 0.5693359450623614, 'person_puts_down_object_on_shelf': 8.586138553443124, 'person_puts_down_object_on_table': 2.2724770752049683, 'person_reads_document': 0.3950204136071002, 'person_rides_bicycle': 1.120112034972008, 'person_shakes_hand': 0.7939816984366945, 'person_sits_down': 0.475769278518198, 'person_stands_up': 0.9511469708058823, 'person_steals_object_from_person': 0.6749376770458883, 'person_talks_on_phone': 0.1437150138699926, 'person_talks_to_person': 0.12953931093794052, 'person_texts_on_phone': 0.34688865531083296, 'person_transfers_object_to_car': 2.2351615915353835, 'person_transfers_object_to_person': 0.6839705972370579, 'person_unloads_car': 0.6080991393803349, 'vehicle': 0.060641247145963084}

        self._class_to_index = {'car_drops_off_person': 0, 'car_picks_up_person': 1, 'car_reverses': 2, 'car_starts': 3, 'car_stops': 4, 'car_turns_left': 5, 'car_turns_right': 6, 'hand_interacts_with_person_highfive': 7, 'person': 8, 'person_abandons_object': 9, 'person_carries_heavy_object': 10, 'person_closes_car_door': 11, 'person_closes_car_trunk': 12, 'person_closes_facility_door': 13, 'person_embraces_person': 14, 'person_enters_car': 15, 'person_enters_scene_through_structure': 16, 'person_exits_car': 17, 'person_exits_scene_through_structure': 18, 'person_holds_hand': 19, 'person_interacts_with_laptop': 20, 'person_loads_car': 21, 'person_opens_car_door': 22, 'person_opens_car_trunk': 23, 'person_opens_facility_door': 24, 'person_picks_up_object_from_floor': 25, 'person_picks_up_object_from_table': 26, 'person_purchases_from_cashier': 27, 'person_purchases_from_machine': 28, 'person_puts_down_object_on_floor': 29, 'person_puts_down_object_on_shelf': 30, 'person_puts_down_object_on_table': 31, 'person_reads_document': 32, 'person_rides_bicycle': 33, 'person_shakes_hand': 34, 'person_sits_down': 35, 'person_stands_up': 36, 'person_steals_object_from_person': 37, 'person_talks_on_phone': 38, 'person_talks_to_person': 39, 'person_texts_on_phone': 40, 'person_transfers_object_to_car': 41, 'person_transfers_object_to_person': 42, 'person_unloads_car': 43, 'vehicle': 44}
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_nb):
        (x,Y) = batch  
        y_hat = self.forward(x)

        (loss, n_valid) = (0, 0)
        C = torch.tensor([self._class_to_weight[k] for (k,v) in sorted(self._class_to_index.items(), key=lambda x: x[1])], device=y_hat.device)  # inverse class frequency        
        for (yh, s) in zip(y_hat, Y):
            labels = json.loads(s)
            if labels is None:
                continue  # skip me
            lbllist = [l for lbl in labels for l in lbl]  # list of multi-labels within clip (unpack from JSON to use default collate_fn)
            lbl_frequency = vipy.util.countby(lbllist, lambda x: x)  # frequency within clip
            lbl_weight = {k:v/float(len(lbllist)) for (k,v) in lbl_frequency.items()}  # multi-label likelihood within clip, sums to one
            for (y,w) in lbl_weight.items():
                # Pick all labels normalized (https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf)
                loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)
            n_valid += 1
        loss = loss / float(max(1, n_valid))  # batch reduction: mean
        return {'loss': loss}
        
    def validation_step(self, batch, batch_nb):
        return {'val_loss': self.training_step(batch, batch_nb)['loss']}

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
    def _totensor(v, training, validation, input_size, num_frames, mean, std, noflip=None, show=False):
        assert isinstance(v, vipy.video.Scene), "Invalid input"

        try:
            v = v.download() if (not v.hasfilename() and v.hasurl()) else v  # fetch it, but do not do this during training!
            maxframes = v.duration_in_frames_of_videofile()
            (i,j) = (v.primary_activity().startframe(), v.primary_activity().endframe())
            startframe = max(0, np.random.randint(i-(num_frames//2), j-(num_frames//2)))
            vc = v.clone().clip(startframe, min(maxframes-1, startframe+num_frames))  # may fail                                 
            vc = vc.trackcrop(dilate=1.2, maxsquare=True)  # may be None

            if training:
                #vc = vc.fliplr() if (np.random.rand() > 0.5 and (noflip is None or vc.category() not in noflip)) else vc
                vc = vc.resize(input_size, input_size)  
            elif validation:
                vc = vc.resize(input_size, input_size)
            else:
                vc = vc.resize(input_size, input_size)

            if show:
                vc.clone().mindim(512).show(timestamp=True)
                b = vc.binarymask().frame(0).rgb().show(figure='binary mask: frame 0')
                
            vc = vc.load().normalize(mean=mean, std=std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load()        
            (t,lbl) = vc.torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw', withlabel=True)  # (c=3)x(d=num_frames)x(H=input_size)x(W=input_size)             
            b = vc.binarymask().bias(-0.5).torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw')  # (c=1)x(d=num_frames)x(H=input_size)x(W=input_size), in [-0.5, 0.5]
            t = torch.cat((t,b), dim=0)  # (c=4) x (d=num_frames) x (H=input_size) x (W=input_size)
        except:
            print('ERROR: %s' % (str(v)))
            t = torch.zeros(4, num_frames, input_size, input_size)  # skip me
            lbl = None
            
        if training or validation:
            return (t, json.dumps(lbl))  # json to use default collate_fn
        else:
            return t

    def totensor(self, v=None, training=False, validation=False, show=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show:
             PIP_250k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right'], show=show))
        return f(v) if v is not None else f
    
