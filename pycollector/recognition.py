import os
import sys
import torch
import vipy
import shutil
import numpy as np
from vipy.util import remkdir, filetail, readlist, tolist, filepath
from datetime import datetime
from pycollector.video import Video
from pycollector.model.yolov3.network import Darknet
from pycollector.globals import print
from pycollector.model.pyvideoresearch.bases.resnet50_3d import ResNet503D, ResNet3D, Bottleneck3D
import pycollector.model.ResNets_3D_PyTorch.resnet
import pycollector.label
import pycollector.dataset
import vipy.activity
import itertools
import copy
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
import math

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
    
    def classlist(self):
        return [k for (k,v) in sorted(list(self.class_to_index().items()), key=lambda x: x[0])]  # sorted in index order

    def num_classes(self):
        return len(self.classlist())

    def fromindex(self, k):
        index_to_class = {v:k for (k,v) in self.class_to_index().items()}
        assert k in index_to_class, "Invalid class index '%s'" % (str(k))
        return index_to_class[k]

    def label_confidence(self, video=None, tensor=None, threshold=None):
        raise
        logits = self.__call__(video, tensor)
        conf = [[(self.index_to_class(j), s[j]) for j in i[::-1] if threshold is None or s[j]>threshold] for (s,i) in zip(logits, np.argsort(logits, axis=1))]
        return conf if len(logits) > 1 else conf[0]

    def activity(self, video, threshold=None):
        (c,s) = zip(*self.label_confidence(video=video, threshold=None))
        return vipy.activity.Activity(startframe=0, endframe=self._num_frames, category=c[0], actorid=video.actorid(), confidence=s[0]) if (threshold is None or s[0]>threshold) else None
            
    def top1(self, video=None, tensor=None, threshold=None):
        raise
        return self.topk(k=1, video=video, tensor=tensor, threshold=threshold)

    def topk(self, k, video=None, tensor=None, threshold=None):
        raise
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
        
    
    
class PIP_250k(pl.LightningModule, ActivityRecognition):
    """Activity recognition using people in public - 250k stabilized"""
    
    def __init__(self, pretrained=True, deterministic=False, modelfile=None, mlbl=False, mlfl=False):
        super().__init__()
        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = mlfl
        self._mlbl = mlbl

        if deterministic:
            np.random.seed(42)

        self._class_to_weight = {'car_drops_off_person': 1.4162811344926518, 'car_picks_up_person': 1.4103618337303332, 'car_reverses': 1.0847976470131024, 'car_starts': 1.0145749063037774, 'car_stops': 0.6659236295324015, 'car_turns_left': 2.942269221156227, 'car_turns_right': 1.1077783089040996, 'hand_interacts_with_person_highfive': 2.793646013249904, 'person': 0.4492053391155403, 'person_abandons_object': 1.0944029463871692, 'person_carries_heavy_object': 0.5848339202761978, 'person_closes_car_door': 0.8616907697519004, 'person_closes_car_trunk': 1.468393359799126, 'person_closes_facility_door': 0.8927495923340439, 'person_embraces_person': 0.6072654081071569, 'person_enters_car': 1.3259274145537951, 'person_enters_scene_through_structure': 0.6928103470838287, 'person_exits_car': 1.6366577285051707, 'person_exits_scene_through_structure': 0.8368692178634396, 'person_holds_hand': 1.2378881634203558, 'person_interacts_with_laptop': 1.6276031281396193, 'person_loads_car': 2.170167410167583, 'person_opens_car_door': 0.7601817241565009, 'person_opens_car_trunk': 1.7255285914206204, 'person_opens_facility_door': 0.9167411017455822, 'person_picks_up_object_from_floor': 1.123251610875369, 'person_picks_up_object_from_table': 3.5979689180114205, 'person_purchases_from_cashier': 7.144918373837205, 'person_purchases_from_machine': 5.920886403645001, 'person_puts_down_object_on_floor': 0.7295795950752353, 'person_puts_down_object_on_shelf': 9.247614426653692, 'person_puts_down_object_on_table': 1.9884672074906158, 'person_reads_document': 0.7940480628992879, 'person_rides_bicycle': 2.662661823600623, 'person_shakes_hand': 0.7819547332927879, 'person_sits_down': 0.8375202893491961, 'person_stands_up': 1.0285510019795079, 'person_steals_object_from_person': 1.0673909796893626, 'person_talks_on_phone': 0.3031855242664589, 'person_talks_to_person': 0.334895684562076, 'person_texts_on_phone': 0.713951043919232, 'person_transfers_object_to_car': 3.2832615561297605, 'person_transfers_object_to_person': 0.9633429807282274, 'person_unloads_car': 1.1051597100801462, 'vehicle': 1.1953172363332243}
        self._class_to_weight['person_puts_down_object_on_shelf'] = 1.0   # run 5

        self._class_to_index = {'car_drops_off_person': 0, 'car_picks_up_person': 1, 'car_reverses': 2, 'car_starts': 3, 'car_stops': 4, 'car_turns_left': 5, 'car_turns_right': 6, 'hand_interacts_with_person_highfive': 7, 'person': 8, 'person_abandons_object': 9, 'person_carries_heavy_object': 10, 'person_closes_car_door': 11, 'person_closes_car_trunk': 12, 'person_closes_facility_door': 13, 'person_embraces_person': 14, 'person_enters_car': 15, 'person_enters_scene_through_structure': 16, 'person_exits_car': 17, 'person_exits_scene_through_structure': 18, 'person_holds_hand': 19, 'person_interacts_with_laptop': 20, 'person_loads_car': 21, 'person_opens_car_door': 22, 'person_opens_car_trunk': 23, 'person_opens_facility_door': 24, 'person_picks_up_object_from_floor': 25, 'person_picks_up_object_from_table': 26, 'person_purchases_from_cashier': 27, 'person_purchases_from_machine': 28, 'person_puts_down_object_on_floor': 29, 'person_puts_down_object_on_shelf': 30, 'person_puts_down_object_on_table': 31, 'person_reads_document': 32, 'person_rides_bicycle': 33, 'person_shakes_hand': 34, 'person_sits_down': 35, 'person_stands_up': 36, 'person_steals_object_from_person': 37, 'person_talks_on_phone': 38, 'person_talks_to_person': 39, 'person_texts_on_phone': 40, 'person_transfers_object_to_car': 41, 'person_transfers_object_to_person': 42, 'person_unloads_car': 43, 'vehicle': 44}

        self._verb_to_noun = {k:set(['car','vehicle','motorcycle','bus','truck']) if (k.startswith('car') or k.startswith('motorcycle') or k.startswith('vehicle')) else set(['person']) for k in self.classlist()}        
        self._class_to_shortlabel = pycollector.label.pip_to_shortlabel

        if pretrained:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())
        elif modelfile is not None:
            self._load_trained(modelfile)
        
    def category(self, x):
        yh = self.forward(x if x.ndim == 5 else torch.unsqueeze(x, 0))
        return [self.index_to_class(int(k)) for (c,k) in zip(*torch.max(yh, dim=1))]

    def category_confidence(self, x):
        yh = self.forward(x if x.ndim == 5 else torch.unsqueeze(x, 0))
        return [(self.index_to_class(int(k)), float(c)) for (c,k) in zip(*torch.max(yh, dim=1))]

    def topk(self, x_logits, k):
        yh = x_logits.detach().cpu().numpy()
        topk = [[(self.index_to_class(j), s[j]) for j in i[-k:][::-1]] for (s,i) in zip(yh, np.argsort(yh, axis=1))]
        return topk

    def topk_probability(self, x_logits, k):
        yh = x_logits.detach().cpu().numpy()
        yh_prob = F.softmax(x_logits, dim=1).detach().cpu().numpy()
        topk = [[(self.index_to_class(j), c[j], p[j]) for j in i[-k:][::-1]] for (c,p,i) in zip(yh, yh_prob, np.argsort(yh, axis=1))]
        return topk
        
    # ---- <LIGHTNING>
    def forward(self, x):
        return self.net(x)  # lighting handles device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_nb, logging=True, valstep=False):
        (x,Y) = batch  
        y_hat = self.forward(x)
        y_hat_softmax = F.softmax(y_hat, dim=1)

        (loss, n_valid) = (0, 0)
        C = torch.tensor([self._class_to_weight[k] for (k,v) in sorted(self._class_to_index.items(), key=lambda x: x[1])], device=y_hat.device)  # inverse class frequency        
        for (yh, yhs, labelstr) in zip(y_hat, y_hat_softmax, Y):
            labels = json.loads(labelstr)
            if labels is None:
                continue  # skip me
            lbllist = [l for lbl in labels for l in lbl]  # list of multi-labels within clip (unpack from JSON to use default collate_fn)
            lbl_frequency = vipy.util.countby(lbllist, lambda x: x)  # frequency within clip
            lbl_weight = {k:v/float(len(lbllist)) for (k,v) in lbl_frequency.items()}  # multi-label likelihood within clip, sums to one
            for (y,w) in lbl_weight.items():
                if valstep:
                    # Pick all labels normalized (https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf
                    loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)
                elif self._mlfl:
                    # Pick all labels normalized, with multi-label focal loss
                    loss += torch.min(torch.tensor(1.0, device=y_hat.device), ((w-yhs[self._class_to_index[y]])/w)**2)*float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)  
                elif self._mlbl:
                    # Pick all labels normalized with multi-label background loss
                    j = self._class_to_index['person'] if (y.startswith('person') or y.startswith('hand')) else self._class_to_index['vehicle']
                    loss += ((1-torch.sqrt(yhs[j]*yhs[self._class_to_index[y]]))**2)*float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)  
                else:
                    # Pick all labels normalized (https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf
                    loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)

            n_valid += 1
        loss = loss / float(max(1, n_valid))  # batch reduction: mean

        if logging:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb, logging=False, valstep=True)['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
    # ---- </LIGHTNING>
    
    @classmethod
    def from_checkpoint(cls, checkpointpath):
        return cls().load_from_checkpoint(checkpointpath)  # lightning
            
    def _load_trained(self, ckptfile):
        self.net = pycollector.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=self.num_classes())
        t = torch.split(self.net.conv1.weight.data, dim=1, split_size_or_sections=1)
        self.net.conv1.weight.data = torch.cat( (*t, t[-1]), dim=1).contiguous()
        self.net.conv1.in_channels = 4  # inflate RGB -> RGBA
        self.load_state_dict(torch.load(ckptfile)['state_dict'])  # FIXME
        self.eval()
        return self
        
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
    def _totensor(v, training, validation, input_size, num_frames, mean, std, noflip=None, show=False, doflip=False):
        assert isinstance(v, vipy.video.Scene), "Invalid input"
        
        try:
            v = v.download() if (not v.hasfilename() and v.hasurl()) else v  # fetch it if necessary, but do not do this during training!        
            if training or validation:
                (ai,aj) = (v.primary_activity().startframe(), v.primary_activity().endframe())  # activity (start,end)
                (ti,tj) = (v.actor().startframe(), v.actor().endframe())  # track (start,end) 
                startframe = np.random.randint(max(0, ti-(num_frames//2)), max(1, tj-(num_frames//2)))  # random startframe that contains track
                endframe = min((startframe+num_frames), aj)  # endframe truncated to be end of activity
                (startframe, endframe) = (startframe, endframe) if (startframe < endframe) else (max(0, aj-num_frames), aj)  # fallback
                assert endframe - startframe <= num_frames
                vc = v.clone().clip(startframe, endframe)    # may fail for some short clips
                vc = vc.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)   
                vc = vc.fliplr() if (doflip or (np.random.rand() > 0.5)) and (noflip is None or vc.category() not in noflip) else vc
            else:
                vc = v.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)  
                vc = vc.fliplr() if doflip and (noflip is None or vc.category() not in noflip) else vc
                
            if show:
                vc.clone().resize(512,512).show(timestamp=True)
                vc.clone().binarymask().frame(0).rgb().show(figure='binary mask: frame 0')
                
            vc = vc.load(shape=(input_size, input_size, 3)).normalize(mean=mean, std=std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load() with known shape
            (t,lbl) = vc.torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw', withlabel=training or validation, nonelabel=True)  # (c=3)x(d=num_frames)x(H=input_size)x(W=input_size), reuses vc._array
            t = torch.cat((t, vc.asfloatmask(fg=0.5, bg=-0.5).torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw')), dim=0)  # (c=4) x (d=num_frames) x (H=input_size) x (W=input_size), copy
            
        except Exception as e:
            if training or validation:
                #print('ERROR: %s' % (str(v)))
                t = torch.zeros(4, num_frames, input_size, input_size)  # skip me
                lbl = None
            else:
                print('WARNING: discarding tensor for video "%s" with exception "%s"' % (str(v), str(e)))
                t = torch.zeros(4, num_frames, input_size, input_size)  # skip me (should never get here)
            
        if training or validation:
            return (t, json.dumps(lbl))  # json to use default collate_fn
        else:
            return t

    def totensor(self, v=None, training=False, validation=False, show=False, doflip=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show:
             PIP_250k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right'], show=show, doflip=doflip))
        return f(v) if v is not None else f
    

class ActivityTracker(PIP_250k):
    def __init__(self, stride=1, activities=None, gpus=None, batchsize=None, mlbl=False, mlfl=False, modelfile=None):
        assert modelfile is not None

        super().__init__(pretrained=False, modelfile=modelfile, mlbl=mlbl, mlfl=mlfl)
        self._stride = stride
        self._allowable_activities = {k:v for (k,v) in [(a,a) if not isinstance(a, tuple) else a for a in activities]} if activities is not None else {k:k for k in self.classlist()}
        self._batchsize_per_gpu = batchsize
        self._gpus = gpus

        if gpus is not None:
            assert torch.cuda.is_available()
            assert batchsize is not None
            self._devices = ['cuda:%d' % k for k in gpus]
            self._gpus = [copy.deepcopy(self.net).to(d, non_blocking=False) for d in self._devices]  
            for m in self._gpus:
                m.eval()
        torch.set_grad_enabled(False)

    def temporal_stride(self, s=None):
        if s is not None:
            self._stride = s
            return self
        else:
            return self._stride

    def forward(self, x):
        """Overload forward for multi-gpu batch.  Don't use torch DataParallel!"""
        if self._gpus is None:
            return super().forward(x)  # cpu
        else:
            x_forward = None            
            for b in x.split(self._batchsize_per_gpu*len(self._gpus)):  # pinned copy
                n_todevice = np.sum(np.array([1 if k<len(b) else 0 for k in range(int(len(self._devices)*np.ceil(len(b)/len(self._devices))))]).reshape(-1, len(self._devices)), axis=0).tolist()
                todevice = [t.pin_memory().to(d, non_blocking=True) for (t,d) in zip(b.split(n_todevice), self._devices) if len(t)>0]   # async device copy
                ondevice = [m(t) for (m,t) in zip(self._gpus, todevice)]   # async
                fromdevice = torch.cat([t.cpu() for t in ondevice], dim=0)
                x_forward = fromdevice if x_forward is None else torch.cat((x_forward, fromdevice), dim=0)
                del ondevice, todevice, fromdevice, b  # force garbage collection of GPU memory
            del x  # force garbage collection
            return x_forward

    def lrt(self, x_logits, lrt_threshold):
        """top-k with likelihood ratio test with background null hypothesis"""
        yh = x_logits.detach().cpu().numpy()
        yh_softmax = F.softmax(x_logits.detach().cpu(), dim=1)        
        p_null = np.maximum(yh[:, self._class_to_index['person']], yh[:, self._class_to_index['vehicle']]).reshape(yh.shape[0], 1)
        lr = yh - p_null   # ~= log likelihood ratio
        f_logistic = lambda x,b,s=1.0: float(1.0 / (1.0 + np.exp(-s*(x + b))))
        return [sorted([(self.index_to_class(j), s[j], t[j], f_logistic(s[j], 1.0)*f_logistic(t[j], 0.0), sm[j]) for j in range(len(s)) if t[j] >= lrt_threshold], key=lambda x: x[3], reverse=True) for (s,t,sm) in zip(yh, lr, yh_softmax)]
        
    def __call__(self, vi, topk=1, activityiou=0, mirror=False, minprob=0, trackconf=0.1, maxdets=None, lr_threshold=-2.0, lr_merge_threshold=-2.0, avgdets=None):
        (n,m) = (self.temporal_support(), self.temporal_stride())
        aa = self._allowable_activities  # dictionary mapping of allowable classified activities to output names        
        f_encode = self.totensor(training=False, validation=False, show=False, doflip=False)  # video -> tensor CxNxHxW
        f_mirror = lambda t: (t, torch.from_numpy(np.copy(np.flip(np.asarray(t), axis=3))))  # CxNxHxW -> CxNxHx(-W), np.flip is much faster than torch.flip, faster than encode mirror=True
        f_totensor = lambda v: (f_encode(v.clone(sharedarray=True)),) if (not mirror or v.actor().category() != 'person') else f_mirror(f_encode(v.clone(sharedarray=True)))
        f_totensorlist = lambda V: [t for v in V for t in f_totensor(v)]        
        def f_reduce(T,V):
            j = sum([v.actor().category() == 'person' for v in V])  # person mirrored, vehicle not mirrored
            (tm, t) = torch.split(T, (2*j, len(T)-2*j), dim=0)  # assumes sorted order, person first, only person/vehicle
            return torch.cat((torch.mean(tm.view(-1, 2, tm.shape[1]), dim=1), t), dim=0) if j>0 else T  # mean over mirror augmentation
        
        try:
            vp = next(vi)  # peek in generator to create clip
            vi = itertools.chain([vp], vi)  # unpeek
            sw = vipy.util.Stopwatch()  # real-time framerate estimate
            framerate = vp.framerate()
            for (k, (v,vc)) in enumerate(zip(vi, vp.stream().clip(n, m, continuous=True, activities=False))):
                videotracks = [] if vc is None else [vt for vt in vc.trackfilter(lambda t: len(t)>=4 and (t.category() == 'person' or (t.category() == 'vehicle' and v.track(t.id()).ismoving(k-10*n, k)))).tracksplit()]  # vehicle moved recently?
                videotracks.sort(key=lambda v: v.actor().confidence(last=1))  # in-place
                numdets = (maxdets if ((avgdets is None) or (sw.duration()<=60) or ((sw.duration()>60) and ((k/sw.duration())/vp.framerate())>0.8)) else
                           (avgdets if ((k/sw.duration())/vp.framerate())>0.67 else int(avgdets//2)))   # real-time throttle schedule
                videotracks = videotracks[-numdets:] if (numdets is not None and len(videotracks)>numdets) else videotracks   # select only the most confident for detection
                videotracks.sort(key=lambda v: v.actor().category())  # in-place, for grouping mirrored encoding: person<vehicle
                
                if len(videotracks)>0 and (k > n):
                    logits = self.forward(torch.stack(f_totensorlist(videotracks))) # augmented logits in track index order, copy
                    logits = f_reduce(logits, videotracks) if mirror else logits  # reduced logits in track index order
                    (actorid, actorcategory) = ([t.actorid() for t in videotracks], [t.actor().category() for t in videotracks])
                    dets = [vipy.activity.Activity(category=aa[category], shortlabel=self._class_to_shortlabel[category], startframe=k-n, endframe=k, confidence=sm, framerate=framerate, actorid=actorid[j], attributes={'pip':category, 'lr':lr}) 
                            for (j, category_conf_lr_prob_sm) in enumerate(self.lrt(logits, lr_threshold))  # likelihood ratio test
                            for (category, conf, lr, prob, sm) in category_conf_lr_prob_sm   
                            if ((category in aa) and   # requested activities only
                                (actorcategory[j] in self._verb_to_noun[category]) and   # noun matching with category renaming dictionary
                                prob>minprob)]   # minimum probability for new activity detection
                    v.assign(k, [d for d in dets if d.attributes['lr'] >= lr_merge_threshold], activityiou=activityiou, activitymerge=True)   # assign new activity detections by merging overlapping activities with weighted average of probability
                    v.assign(k, [d for d in dets if d.attributes['lr'] < lr_merge_threshold], activityiou=activityiou, activitymerge=False)   # create new activity detections for extremely low confidence detections 
                    del logits, dets, videotracks  # torch garabage collection
                yield v

        except Exception as e:                
            raise

        finally:
            # Activity probability:  noun*verb_proposal*verb probability
            nounconf = {k:t.confidence(samples=8) for (k,t) in v.tracks().items()}
            v.activitymap(lambda a: a.confidence(nounconf[a.actorid()]*a.confidence()))
            
            # Bad tracks:  Remove low confidence or too short non-moving tracks
            v.trackfilter(lambda t: len(t)>=v.framerate() and (t.confidence() >= trackconf or t.startbox().iou(t.endbox()) == 0)).activityfilter(lambda a: a.actorid() in v.tracks())  
            
            # Missing objects:  Significantly reduce confidence of complex classes (yuck)
            v.activitymap(lambda a: a.confidence(0.01*a.confidence()) if (a.category() in ['person_steals_object', 'person_abandons_package', 'person_purchases']) else a) 

            # Missing objects:  Reduce confidence of classes without an associated object detection
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() in ['person_talks_on_phone', 'person_texts_on_phone', 'person_reads_document', 'person_interacts_with_laptop', 'person_carries_heavy_object']) else a) 
            
            # Vehicle track:  High confidence vehicle turns must be a minimum angle
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if ((a.category() in ['vehicle_turns_left', 'vehicle_turns_right']) and (abs(v.track(a.actorid()).bearing_change(a.startframe(), a.endframe(), dt=v.framerate())) < (np.pi/4))) else a)

            # Vehicle track:  U-turn can only be distinguished from left/right turn at the end of a track by looking at the turn angle
            v.activitymap(lambda a: a.category('vehicle_makes_u_turn').shortlabel('u turn') if ((a.category() in ['vehicle_turns_left', 'vehicle_turns_right']) and (abs(v.track(a.actorid()).bearing_change(a.startframe(), a.endframe(), dt=v.framerate())) > (np.pi-(np.pi/4)))) else a)

            # Vehicle track: Starts and stops must be at most 5 seconds (yuck)
            v.activitymap(lambda a: a.truncate(a.startframe(), a.startframe()+v.framerate()*5) if a.category() in ['vehicle_starts', 'vehicle_reverses'] else a)
            v.activitymap(lambda a: a.truncate(a.endframe()-5*v.framerate(), a.endframe()) if a.category() == 'vehicle_stops' else a)
            v.activitymap(lambda a: a.truncate(a.middleframe()-2.5*v.framerate(), a.middleframe()+2.5*v.framerate()) if a.category() in ['vehicle_turns_left', 'vehicle_turns_right', 'vehicle_makes_u_turn'] else a)   

            # Group activity: Must be accompanied by a friend with the same activity detection
            dstbox = {k:v.track(a.actorid()).boundingbox(a.startframe(), a.endframe()) for (k,a) in v.activities().items()}  # precompute
            srcbox = {k:bb.clone().maxsquare().dilate(1.2) for (k,bb) in dstbox.items()}                            
            for c in ['person_embraces_person', 'hand_interacts_with_person', 'person_talks_to_person', 'person_transfers_object']:
                v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() == c and
                                                                              not any([(af.category() == c and
                                                                                        af.id() != a.id() and
                                                                                        af.actorid() != a.actorid() and 
                                                                                        af.during_interval(a.startframe(), a.endframe(), inclusive=True) and 
                                                                                        srcbox[a.id()].iou(dstbox[af.id()]) > 0)
                                                                                       for af in v.activities().values()])) else a)
            
            # Person/Bicycle track: riding must be accompanied by an associated moving bicycle track
            v.activityfilter(lambda a: a.category() != 'person_rides_bicycle')
            bikelist = [v.add(vipy.activity.Activity(startframe=t.startframe(), endframe=t.endframe(), category='person_rides_bicycle', shortlabel='rides', confidence=t.confidence(samples=8), framerate=v.framerate(), actorid=t.id(), attributes={'pip':'person_rides_bicycle', 'lr':lr_merge_threshold}))
                        for (tk,t) in v.tracks().items() if t.category() == 'bicycle' and t.ismoving()]

            # Person/Vehicle track: person/vehicle interaction must be accompanied by an associated stopped vehicle track
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if ((a.category().startswith('person') and ('vehicle' in a.category() or 'trunk' in a.category())) and not any([t.category() == 'vehicle' and
                                                                                                                                                                                     t.segment_maxiou(v.track(a.actorid()), a.startframe(), a.endframe()) > 0 and
                                                                                                                                                                                     not t.ismoving(a.startframe(), a.endframe())
                                                                                                                                                                                     for t in v.tracks().values()])) else a)
            # Vehicle/Person track: vehicle/person interaction must be accompanied by an associated person track
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if ((a.category().startswith('vehicle') and ('person' in a.category())) and not any([t.category() == 'person' and t.segment_maxiou(v.track(a.actorid()), a.startframe(), a.endframe()) > 0 for t in v.tracks().values()])) else a)

            # Vehicle/Person track: vehicle dropoff/pickup must be accompanied by an associated person track start/end
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() == 'vehicle_drops_off_person' and not any([t.category() == 'person' and t.segment_maxiou(v.track(a.actorid()), t.startframe(), t.startframe()+1) > 0 for t in v.tracks().values()])) else a)
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() == 'vehicle_picks_up_person' and not any([t.category() == 'person' and t.segment_maxiou(v.track(a.actorid()), t.endframe()-1, t.endframe()) > 0 for t in v.tracks().values()])) else a)
            
            # Person track: enter/exit scene cannot be at the image boundary
            boundary = v.framebox().dilate(0.9)
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() == 'person_enters_scene_through_structure' and v.track(a.actorid())[max(a.startframe(), v.track(a.actorid()).startframe())].cover(boundary) < 1) else a)
            v.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.category() == 'person_exits_scene_through_structure' and v.track(a.actorid())[min(a.endframe(), v.track(a.actorid()).endframe())].cover(boundary) < 1) else a)
                        
            # Activity union:  Temporal gaps less than support should be merged into one activity detection for a single track
            merged = set([])
            mergeable_dets = [a for a in v.activities().values() if a.attributes['lr'] >= lr_merge_threshold]  # only mergeable detections
            other = sorted(mergeable_dets, key=lambda a: a.startframe())  # other, sort once
            for a in sorted(mergeable_dets, key=lambda a: a.startframe()):
                for o in other:
                    if (o.startframe() >= a.startframe()) and (a.id() != o.id()) and (o.actorid() == a.actorid()) and (o.category() == a.category()) and (o.id() not in merged) and (a.id() not in merged) and (a.temporal_distance(o) <= self.temporal_support()): 
                        a.union(o)  # in-place update
                        merged.add(o.id())
            v.activityfilter(lambda a: a.id() not in merged)

            # Activity union:  "Brief" breaks of these activities should be merged into one activity detection for a single track
            tomerge = set(['person_reads_document', 'person_interacts_with_laptop', 'person_talks_to_person', 'person_purchases', 'person_steals_object', 'person_talks_on_phone', 'person_texts_on_phone', 'person_rides_bicycle', 'person_carries_heavy_object', 'person', 'vehicle'])
            merged = set([])
            other = sorted(mergeable_dets, key=lambda a: a.startframe())  # other            
            for a in sorted(mergeable_dets, key=lambda a: a.startframe()):
                if a.category() in tomerge:
                    for o in other:
                        if (o.startframe() >= a.startframe()) and (o.id() != a.id()) and (o.actorid() == a.actorid()) and (o.category() == a.category()) and (o.id() not in merged) and (a.id() not in merged) and (a.temporal_distance(o) < 10*v.framerate()):  # "brief" == "<10s"
                            a.union(o)  # in-place update
                            merged.add(o.id())
            v.activityfilter(lambda a: a.id() not in merged)            

            # Activity group suppression:  Group activities may have at most one activity detection of this type per group in a spatial region surrounding the actor
            tosuppress = set(['hand_interacts_with_person', 'person_embraces_person', 'person_transfers_object', 'person_steals_object', 'person_purchases', 'person_talks_to_person'])
            suppressed = set([])
            for a in sorted(v.activities().values(), key=lambda a: a.confidence(), reverse=True):  # decreasing confidence
                if a.category() in tosuppress:
                    for o in v.activities().values():  # other activities
                        if (o.actorid() != a.actorid() and  # different tracks
                            o.category() == a.category() and  # same category
                            o.confidence() <= a.confidence() and   # lower confidence
                            o.id() not in suppressed and  # not already suppressed
                            o.during_interval(a.startframe(), a.endframe()) and # overlaps temporally by at least one frame
                            (v.track(a.actorid()).boundingbox(a.startframe(), a.endframe()) is not None and v.track(o.actorid()).boundingbox(a.startframe(), a.endframe()) is not None) and   # has valid tracks
                            (v.track(a.actorid()).boundingbox(a.startframe(), a.endframe()).dilate(1.2).maxsquare().iou(v.track(o.actorid()).boundingbox(a.startframe(), a.endframe()).dilate(1.2).maxsquare()) > 0)):  # overlaps spatially "close by"
                            suppressed.add(o.id())  # greedy non-maximum suppression of lower confidence activity detection
            v.activityfilter(lambda a: a.id() not in suppressed)
            v.setattribute('_completed', str(datetime.now()))

