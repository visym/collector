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
        self.net =  None
        self._class_to_index = {}
        self._num_frames = 0
        
    def class_to_index(self):
        return self._class_to_index
    
    def index_to_class(self, index=None):
        d = {v:k for (k,v) in self.class_to_index().items()}
        return d if index is None else d[index]
    
    def __call__(self, video=None, tensor=None):
        assert self.net is not None, "Invalid network"
        assert video is not None or tensor is not None
        return self.net(tensor)

    def classlist(self):
        return [k for (k,v) in sorted(list(self.class_to_index().items()), lambda k,v: v)]

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
        d_pip_to_meva = {'person_closes_car_door': 'person_closes_vehicle_door',
                         'person_enters_car': 'person_enters_vehicle',
                         'person_exits_car': 'person_exits_vehicle',
                         'person_opens_car_door': 'person_opens_vehicle_door',
                         'person_closes_car_trunk': 'person_closes_trunk',
                         'person_loads_car': 'person_loads_vehicle',
                         'person_opens_car_trunk': 'person_opens_trunk',
                         'person_unloads_car': 'person_unloads_vehicle',
                         'car_turns_left': 'vehicle_turns_left',
                         'car_turns_right': 'vehicle_turns_right',
                         'car_makes_u_turn': 'vehicle_makes_u_turn',
                         'car_picks_up_person': 'vehicle_picks_up_person',
                         'car_starts': 'vehicle_starts',
                         'car_stops': 'vehicle_stops',
                         'car_drops_off_person': 'vehicle_drops_off_person',
                         'person_transfers_object_to_car': 'person_transfers_object',
                         'car_reverses': 'vehicle_reverses',
                         'person_enters_motorcycle': 'person_enters_vehicle',
                         'person_exits_motorcycle': 'person_exits_vehicle',
                         'person_closes_motorcycle_trunk': 'person_closes_trunk',
                         'person_loads_motorcycle': 'person_loads_vehicle',
                         'person_opens_motorcycle_trunk': 'person_opens_trunk',
                         'person_unloads_motorcycle': 'person_unloads_vehicle',
                         'motorcycle_turns_left': 'vehicle_turns_left',
                         'motorcycle_turns_right': 'vehicle_turns_right',
                         'motorcycle_makes_u_turn': 'vehicle_makes_u_turn',
                         'motorcycle_picks_up_person': 'vehicle_picks_up_person',
                         'motorcycle_starts': 'vehicle_starts',
                         'motorcycle_stops': 'vehicle_stops',
                         'motorcycle_drops_off_person': 'vehicle_drops_off_person',
                         'person_transfers_object_to_motorcycle': 'person_transfers_object',
                         'motorcycle_reverses': 'vehicle_reverses',
                         'person_closes_facility_door': 'person_closes_facility_door',
                         'person_enters_scene_through_structure': 'person_enters_scene_through_structure',
                         'person_comes_into_scene_through_structure': 'person_enters_scene_through_structure',
                         'person_exits_scene_through_structure': 'person_exits_scene_through_structure',
                         'person_leaves_scene_through_structure': 'person_exits_scene_through_structure',
                         'person_opens_facility_door': 'person_opens_facility_door',
                         'person_reads_document': 'person_reads_document',
                         'person_sits_down': 'person_sits_down',
                         'person_stands_up': 'person_stands_up',
                         'person_puts_down_object': 'person_puts_down_object',
                         'person_picks_up_object': 'person_picks_up_object',
                         'person_picks_up_object_from_table': 'person_picks_up_object',
                         'person_picks_up_object_from_floor': 'person_picks_up_object',
                         'person_picks_up_object_from_shelf': 'person_picks_up_object',
                         'person_puts_down_object_on_table': 'person_puts_down_object',
                         'person_puts_down_object_on_floor': 'person_puts_down_object',
                         'person_puts_down_object_on_shelf': 'person_puts_down_object',
                         'person_abandons_package': 'person_abandons_package',
                         'person_abandons_bag': 'person_abandons_package',
                         'person_carries_heavy_object': 'person_carries_heavy_object',
                         'person_talks_on_phone': 'person_talks_on_phone',
                         'person_texts_on_phone': 'person_texts_on_phone',
                         'person_interacts_with_laptop': 'person_interacts_with_laptop',
                         'person_purchases_from_machine': 'person_purchases',
                         'person_rides_bicycle': 'person_rides_bicycle',
                         'person_talks_to_person': 'person_talks_to_person',
                         'hand_interacts_with_person_highfive': 'hand_interacts_with_person',
                         'person_steals_object': 'person_steals_object',
                         'person_steals_object_from_person': 'person_steals_object',
                         'person_purchases_from_cashier': 'person_purchases',
                         'person_transfers_object_to_person': 'person_transfers_object',
                         'person_embraces_person': 'person_embraces_person',
                         'hand_interacts_with_person_holdhands': 'hand_interacts_with_person',
                         'hand_interacts_with_person_shakehands': 'hand_interacts_with_person',
                         'person_shakes_hand': 'hand_interacts_with_person',
                         'person_holds_hand': 'hand_interacts_with_person'}
            
        return d_pip_to_meva[cls]

    
