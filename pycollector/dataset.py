import os
import numpy as np
from pycollector.globals import print
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv, newpathroot, listjson, extlist, filefull, tempdir
import random
import vipy
import vipy.util
import shutil
import uuid
import warnings
import copy 
import atexit
from pycollector.util import is_email_address
import torch
from vipy.batch import Batch       
import hashlib
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import pickle
import time
import json
import dill
import vipy.torch
import vipy.dataset

    
def disjoint_activities(V, activitylist):    
    assert all([isinstance(v, vipy.video.Video) for v in V])
    assert isinstance(activitylist, list) and len(activitylist)>0 and len(activitylist[0]) == 2
    for (after, before) in activitylist:
        V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == after]) if a.category() == before else a) for v in V]  
    V = [v.activityfilter(lambda a: len(a)>0) for v in V]  # some activities may be zero length after disjoint
    return V


def asmeva(V):
    """Convert a list of collector.dataset.Video() to MEVA annotation style"""
    assert all([isinstance(v, vipy.video.Scene) for v in V])
    
    # MEVA annotations assumptions:  https://docs.google.com/spreadsheets/d/19I3C5Zb6RHS0QC30nFT_m0ymArzjvlPLfb5SSRQYLUQ/edit#gid=0
    # Pad one second before, zero seconds after
    before1after0 = set(['person_opens_facility_door', 'person_closes_facility_door', 'person_opens_car_door', 'person_closes_car_door', 
                         'person_opens_car_trunk', 'person_opens_motorcycle_trunk', 'person_closes_car_trunk', 'person_closes_motorcycle_trunk',
                         'car_stops', 'motorcycle_stops', 'person_interacts_with_laptop'])        
    
    V = [v.activitymap(lambda a: a.temporalpad( (v.framerate()*1.0, 0) ) if a.category() in before1after0 else a) for v in V]
    
    # pad one second before, one second after, up to maximum of two seconds
    before1after1max2 = set(['person_enters_scene_through_structure'])
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*1.0)-len(a))) if a.category() in before1after1max2 else a) for v in V]
    
    # person_exits_scene_through_structure:  Pad one second before person_opens_facility_door label (if door collection), and ends with enough padding to make this minimum two seconds
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_facility_door'] + [a.startframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_exits_scene_through_structure' else a)) for v in V]        
    
    # person_enters_vehicle: Starts one second before person_opens_vehicle_door activity label and ends at the end of person_closes_vehicle_door activity, split motorcycles into separate class
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_car_door'] + [a.startframe()]))
                                  .endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_door'] + [a.endframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_enters_car' else a)) for v in V]        
    
    # person_exits_vehicle:  Starts one second before person_opens_vehicle_door, and ends at person_exits_vehicle with enough padding to make this minimum two seconds, split motorcycles into separate class
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_car_door'] + [a.startframe()]))
                                  .endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_door'] + [a.endframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_exits_car' else a)) for v in V]        
    
    # person_unloads_vehicle:  No padding before label start (the definition states one second of padding before cargo starts to move, but our label starts after the trunk is open, 
    # so there is a lag from opening to touching the cargo which we assume is at least 1sec), ends at the end of person_closes_trunk.
    V = [v.activitymap(lambda a: (a.endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_trunk'] + [a.endframe()]))
                                  if a.category() == 'person_unloads_car' else a)) for v in V]
    V = [v.activitymap(lambda a: (a.endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_motorcycle_trunk'] + [a.endframe()]))
                                  if a.category() == 'person_unloads_motorcycle' else a)) for v in V]
    
    # person_talks_to_person:  Equal padding to minimum of five seconds
    equal5 = set(['person_talks_to_person', 'person_reads_document'])
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*2.5)-len(a))) if a.category() in equal5 else a) for v in V]
    
    # person_texting_on_phone:  Equal padding to minimum of two seconds
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*1.0)-len(a))) if a.category() == 'person_texting_on_phone' else a) for v in V]
    
    # Pad one second before, one second after
    before1after1 = set(['car_turns_left', 'motorcycle_turns_left', 'car_turns_right', 'motorcycle_turns_right', 'person_transfers_object_to_person', 'person_transfers_object_to_vehicle',
                         'person_sets_down_object', 'hand_interacts_with_person_handshake', 'hand_interacts_with_person_highfive', 'hand_interacts_with_person_holdhands', 'person_embraces_person', 'person_purchases',
                         'car_picks_up_person', 'car_drops_off_person', 'motorcycle_drops_off_person', 'motorcycle_picks_up_person'])
    V = [v.activitymap(lambda a: a.temporalpad(v.framerate()*1.0) if a.category() in before1after1 else a) for v in V]
    
    # Pad zero second before, one second after
    before0after1 = set(['car_makes_u_turn', 'motorcycle_makes_u_turn', 'person_picks_up_object'])  
    V = [v.activitymap(lambda a: a.temporalpad( (0, v.framerate()*1.0) ) if a.category() in before0after1 else a) for v in V]
    
    # person_abandons_package:  two seconds before, two seconds after
    V = [v.activitymap(lambda a: a.temporalpad(v.framerate()*2.0) if a.category() == 'person_abandons_package' else a) for v in V]
    return V




class TorchDataset(vipy.torch.TorchDataset):
    """Moved to vipy.torch.TorchDataset"""
    # vipy-1.11.7    
    pass

class TorchTensordir(vipy.torch.TorchTensordir):
    """Moved to vipy.torch.TorchTensordir"""
    # vipy-1.11.7    
    pass

class Dataset(vipy.dataset.Dataset):
    """Additional methods beyond `vipy.dataset.Dataset` that are specific to collector style datasets"""

    def collectors(self, outfile=None):
        assert self._isvipy()
        d = vipy.util.countby(self.list(), lambda v: v.attributes['collector_id'])
        f = lambda x,n: len([k for (k,v) in d.items() if int(v) >= n])
        print('[vipy.dataset]: Collectors = %d ' % f(d,0))
        print('[vipy.dataset]: Collectors with >10 submissions = %d' % f(d,10))
        print('[vipy.dataset]: Collectors with >100 submissions = %d' % f(d,100))
        print('[vipy.dataset]: Collectors with >1000 submissions = %d' % f(d,1000))
        print('[vipy.dataset]: Collectors with >10000 submissions = %d' % f(d,10000))

        if outfile is not None:
            from vipy.metrics import histogram
            (k,v) = zip(*(sorted(d.items(), key=lambda x: x[1], reverse=True))) 
            histogram(v, list(range(len(k))), outfile=outfile, ylabel='Submissions', xlabel='Collector', xrot='vertical', fontsize=3, xshow=False)            
        return d

    def os(self, outfile=None):
        assert self._isvipy()
        d = vipy.util.countby([v for v in self.list() if v.hasattribute('device_identifier')], lambda v: v.attributes['device_identifier'])
        print('[vipy.dataset]: Device OS = %d ' % len(d))
        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), d.keys(), explode=None, outfile=outfile,  shadow=False)
        return d

    def device(self, outfile=None, n=24, fontsize=7):
        assert self._isvipy()
        d_all = vipy.util.countby([v for v in self.list() if v.hasattribute('device_type') and v.attributes['device_type'] != 'unrecognized'], lambda v: v.attributes['device_type'])
        
        topk = [k for (k,v) in sorted(list(d_all.items()), key=lambda x: x[1])[-n:]] 
        other = np.sum([v for (k,v) in d_all.items() if k not in set(topk)])

        d = {k:v for (k,v) in d_all.items() if k in set(topk)}
        d.update( {'Other':other} )
        d = dict(sorted(list(d.items()), key=lambda x: x[1]))

        print('[vipy.dataset.device]: Device types = %d ' % len(d_all))
        print('[vipy.dataset.device]: Top-%d Device types = %s ' % (n, str(topk)))

        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), d.keys(), explode=None, outfile=outfile,  shadow=False, legend=False, fontsize=fontsize, rotatelabels=False)
        return d
        

    def geolocation(self, outfile=None):
        assert self._isvipy()
        
        L = [v for v in self if 'ipAddress' in v.metadata()]
        L = [v[0].geolocation() for (k,v) in vipy.util.groupbyasdict(L, lambda v: v.collectorid()).items()]
        L = [v for v in L if v is not None]
        return {'country': vipy.util.countby(L, lambda d: str(d['countryname'])),
                'city': vipy.util.countby(L, lambda d: str(d['city']))}
                
    @classmethod
    def cast(cls, D):
        assert isinstance(D, vipy.dataset.Dataset), "Invalid input - must be derived from vipy.dataset.Dataset"
        return cls(list(D))
        
