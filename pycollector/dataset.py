import os
import numpy as np
import pycollector.detection
from pycollector.globals import print
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv, newpathroot, listjson, extlist, filefull, tempdir
import random
import vipy
import vipy.util
import shutil
import uuid
from pathlib import PurePath
import warnings
import copy 
import atexit
from pycollector.util import is_email_address
import torch
from vipy.batch import Batch       
import hashlib
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import bz2
import pickle
import time

    
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



class Datasets():
    """pycollector.dataset.Datasets() class
    
       This class is designed to transform datasets with intermediate datasets saved in indir
    """

    def __init__(self, indir, strict=False):
        self._indir = remkdir(os.path.abspath(os.path.expanduser(indir)))
        self._archive_ext = ['pkl', 'json']
        self._strict = strict
        self._datasets = {}

        # FIXME: this expression is broken and will not work in general
        self._schema = (lambda dstdir, v, k=None, indir=self._indir, ext=None, category=None, newfile=True: os.path.join(indir, dstdir, 
                                                                                                                         v.category() if category is None else category,
                                                                                                                         (('%s%s.%s' % (v.attributes['video_id'] if v.hasattribute('video_id') else filebase(v.filename()),
                                                                                                                                       ('_%s' % str(k)) if k is not None else '', 
                                                                                                                                       (fileext(v.filename(), withdot=False) if ext is None else str(ext))))
                                                                                                                         if newfile is True else filetail(v.filename()))))
        
        assert os.path.isdir(self._indir), "invalid input directory"

    def __getitem__(self, k):
        return self.load(k)

    def __setitem__(self, k, v):
        self.cache(k, v)

    def __repr__(self):
        return str('<pycollector.datasets: "%s">' % self._indir)

    def list(self):
        return sorted(list(set([filebase(f) for e in self._archive_ext for f in listext(self._indir, '.%s' % e)])))

    def isdataset(self, src):
        return (src in self.list()) or isinstance(src, Dataset) or src in self._datasets
        
    def reload(self, src, format='json'):
        assert self.isdataset(src), "Invalid dataset '%s'" % (str(src))
        loadfile = os.path.join(self._indir, '%s.%s' % (src, format))
        print('[pycollector.datasets]: Loading "%s" ...' % loadfile)
        D = Dataset(vipy.util.load(loadfile, abspath=True), id=src)
        self._datasets[src] = D
        return D
        
    def load(self, src, format='json'):
        if isinstance(src, Dataset):
            return src
        elif src in self._datasets:
            return self._datasets[src]
        else:
            return self.reload(src, format)

    def new(self, objlist, id=None):
        """Create a new dataset, cache it and return Dataset()"""
        d = Dataset(objlist, id=id)
        self.cache(d)
        return d

    def cached(self):
        return list(self._datasets.keys())

    def cache(self, dst):
        D = self.load(dst)
        self._datasets[D.id()] = D
        return self

    def flush(self, dst):
        self._datasets.pop(dst, None)
        return self

    def save(self, dst, format='json', nourl=False, castas=None, relpath=False, noadmin=False, strict=True, significant_digits=2, noemail=True, flush=True):
        self.load(dst).save(os.path.join(self._indir, '%s.%s' % (self.load(dst).id(), format)), nourl=nourl, castas=castas, relpath=relpath, noadmin=noadmin, strict=strict, significant_digits=significant_digits, noemail=noemail, flush=flush)
        return self
    
    def map(self, src, f_transform, model=None, dst=None, ascompleted=True):        
        D = self.load(src)
        assert isinstance(D, Dataset), "Source dataset '%s' does not exist" % src                
        return D.map(f_transform, model=model, dst=dst, checkpoint=False, strict=self._strict, ascompleted=ascompleted)

    def fetch(self, src):
        D = self.load(src)
        assert D.isvipy() and all([v.hasattribute('video_id') and v.category() is not None for v in self.list()]), "Invalid dataset"

        f_saveas = lambda v, outdir=os.path.join(self._indir, src): os.path.join(outdir, v.category(), '%s.%s' % (v.attributes['video_id'], fileext(v.filename(), withdot=False)))
        f_fetch = lambda v, f=f_saveas: v.filename(f(v)).download().print()
        return self.map(src, f_fetch, dst=None)
        
    def track(self, src, dst=None, batchsize=16, conf=0.05, iou=0.5, maxhistory=5, smoothing=None, objects=None, mincover=0.8, maxconf=0.2):
        model = pycollector.detection.VideoTracker(batchsize=batchsize)
        f_track = lambda net,v,b=batchsize: net.gpu(list(range(torch.cuda.device_count())), batchsize=b*torch.cuda.device_count()).track(v, conf, iou, maxhistory, smoothing, objects, mincover, maxconf).print()
        return self.map(src, f_track, model=model, dst=dst)

    def actor_association(self, src, d_category_to_object, batchsize, dst=None, cpu=True, dt=10):
        model = pycollector.detection.ActorAssociation(batchsize=batchsize)
        if cpu:
            f = lambda net,v,dt=dt,d_category_to_object=d_category_to_object: net(v, d_category_to_object[v.category()], dt=dt) if v.category() in d_category_to_object else v
        else:
            f = lambda net,v,b=batchsize: net.gpu(list(range(torch.cuda.device_count())), batchsize=b*torch.cuda.device_count())(v, d_category_to_object[v.category()]) if v.category() in d_category_to_object else v
        return self.map(src, f, model=model, dst=dst)

    def instance_mining(self, src, dstdir=None, dst=None, batchsize=1, minconf=0.01, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05):
        model = pycollector.detection.MultiscaleVideoTracker(batchsize=batchsize)
        dst = dst if dst is not None else '%s_instancemining' % (self.load(src).id())
        dstdir = remkdir(os.path.join(self._indir, dst))
        f_process = lambda net,v,o=objects,dstdir=dstdir: net.gpu(list(range(torch.cuda.device_count()))).track(v, objects=o, verbose=False, minconf=minconf, miniou=miniou, maxhistory=maxhistory, smoothing=smoothing, trackconf=trackconf).pkl(os.path.join(dstdir, '%s.pkl' % v.videoid())).print()
        return self.map(src, f_process, model=model, dst=dst)  

    def stabilize_refine_activityclip(self, src, dst, batchsize=1, dt=5, minlength=5, maxsize=512*3):
        from vipy.flow import Flow
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  
        f_process = (lambda net,v,dt=dt,f=self._schema,dst=dst,maxsize=maxsize: 
                     [a.saveas(f(dst,a,k)).pkl().print() for (k,a) in enumerate(net(Flow(flowdim=256).stabilize(v, strict=False, residual=True, maxsize=maxsize).saveas(f(dst,v), flush=True).pkl().print(),
                                                                                    proposalconf=5E-2, 
                                                                                    proposaliou=0.8, 
                                                                                    miniou=0.2, 
                                                                                    dt=dt, 
                                                                                    mincover=0.8, 
                                                                                    byclass=True, 
                                                                                    shapeiou=0.7, 
                                                                                    smoothing='spline', 
                                                                                    splinefactor=None, 
                                                                                    strict=True).print().pkl(f(dst,v,'refined',ext='pkl')).activityclip())
                      if not a.hasattribute('unstabilized') and not a.hasattribute('unrefined')])

        V = self.map(src, dst, f_process, model=model, save=False).dataset(dst)
        V = [v for clips in V for v in clips if (v is not None) and not v.hasattribute('unrefined') and not v.hasattribute('unstabilized')]  # unpack clips, remove videos that failed
        V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
        return self.new(V, dst)

    def refine_activityclip_stabilize(self, src, dst, batchsize=1, dt=3, minlength=5, padwidthfrac=1.0, padheightfrac=0.2):
        """Refine the bounding box, split into clips then stabilzie the clips.  This is more memory efficient for stabilization"""
        assert self.has_dataset(src) and self.isloaded(src)

        from vipy.flow import Flow
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  
        f_process = (lambda net,v,dt=dt,f=self._schema,dst=dst,padwidthfrac=padwidthfrac,padheightfrac=padheightfrac: 
                     [Flow(flowdim=256).stabilize(a, strict=False, residual=True, padwidthfrac=padwidthfrac, padheightfrac=padheightfrac, maxsize=None).saveas(f(dst,a,k)).pkl().print()
                      for (k,a) in enumerate(net(v,
                                                 proposalconf=5E-2, 
                                                 proposaliou=0.8, 
                                                 miniou=0.2, 
                                                 dt=dt, 
                                                 mincover=0.8, 
                                                 byclass=True, 
                                                 shapeiou=0.7, 
                                                 smoothing='spline', 
                                                 splinefactor=None, 
                                                 strict=True).print().pkl(f(dst,v,'refined',ext='pkl')).activityclip())
                      if a.hastracks() and not a.hasattribute('unrefined')])

        V = self.map(src, dst, f_process, model=model, save=False).dataset(dst)
        V = [v for clips in V for v in clips if (v is not None) and not v.hasattribute('unrefined') and not v.hasattribute('unstabilized')]  # unpack clips, remove videos that failed
        V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) if minlength is not None else v for v in V]  # get rid of activities without tracks greater than dt
        return self.new(V, dst)

    def refine_stabilize(self, src, dst, batchsize=1, dt=3, minlength=5, padwidthfrac=1.0, padheightfrac=0.2):
        """Refine the bounding box, split into clips then stabilzie the clips.  This is more memory efficient for stabilization"""


        self._schema = (lambda dstdir, v, k=None, indir=self._indir, ext=None, category=None: os.path.join(indir, dstdir, 
                                                                                                           v.category() if category is None else category,
                                                                                                           ('%s%s.%s' % (v.attributes['video_id'] if v.hasattribute('video_id') else filebase(v.filename()),
                                                                                                                         ('_%s' % str(k)) if k is not None else '', 
                                                                                                                         (fileext(v.filename(), withdot=False) if ext is None else str(ext))))))


        assert self.has_dataset(src) and self.isloaded(src)

        from vipy.flow import Flow
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize) 
        f_process = (lambda net,v,dt=dt,f=self._schema,dst=dst,padwidthfrac=padwidthfrac,padheightfrac=padheightfrac: 
                     Flow(flowdim=256).stabilize(net(v,
                                                     proposalconf=5E-2, 
                                                     proposaliou=0.8, 
                                                     miniou=0.2, 
                                                     dt=dt, 
                                                     mincover=0.8, 
                                                     byclass=True, 
                                                     shapeiou=0.7, 
                                                     smoothing='spline', 
                                                     splinefactor=None, 
                                                     strict=True).print().pkl(f(dst,v,'refined',ext='pkl')),
                                                 strict=False, residual=True, padwidthfrac=padwidthfrac, padheightfrac=padheightfrac, maxsize=None).saveas(f(dst,v,'stabilized')).pkl().print())

        V = self.map(src, dst, f_process, model=model, save=False).dataset(dst)
        V = [v for v in V if (v is not None) and not v.hasattribute('unrefined') and not v.hasattribute('unstabilized')]  # remove videos that failed
        return self.new(V, dst)
    
    def refine_activityclip(self, src, dst, batchsize=1, dt=5, minlength=5):
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize) 
        f_process = (lambda net,v,dt=dt,f=self._schema,dst=dst: 
                     [a.saveas(f(dst,a,k)).pkl().print() for (k,a) in enumerate(net(v,
                                                                                    proposalconf=5E-2, 
                                                                                    proposaliou=0.8, 
                                                                                    miniou=0.2, 
                                                                                    dt=dt, 
                                                                                    mincover=0.8, 
                                                                                    byclass=True, 
                                                                                    shapeiou=0.7, 
                                                                                    smoothing='spline', 
                                                                                    splinefactor=None, 
                                                                                    strict=True).print().pkl(f(dst,v,'refined',ext='pkl')).activityclip())
                      if not a.hasattribute('unrefined')])

        V = self.map(src, dst, f_process, model=model, save=False).dataset(dst)
        V = [v for clips in V for v in clips if (v is not None) and not v.hasattribute('unrefined')]  # unpack clips, remove videos that failed
        V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
        return self.new(V,dst)

    def pad_activityclip(self, src, dst, t=2):        
        f_tubelet = lambda v, t=t, f=self._schema, dst=dst: [a.saveas(f(dst,a,k)).print() for (k,a) in enumerate(v.activitymap(lambda x: x.padto(t)).activityclip())]
        V = [a for V in self.map(src, dst, f_tubelet, save=False).dataset(dst) for a in V if a is not None]  # unpack
        return self.new(V, dst)

    def stabilize(self, src, dst):
        from vipy.flow import Flow
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_stabilize = (lambda v, f=f_saveas: Flow(flowdim=256).stabilize(v, strict=False, residual=True).saveas(f(v), flush=True).print() if v.canload() else None)
        return self.map(src, dst, f_stabilize)

    def refine(self, src, dst, batchsize=1, dt=3, minlength=5):        
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize) 
        f = lambda net, v, dt=dt: net(v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=dt, mincover=0.8, byclass=True, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True).print()
        V = self.map(src, dst, f, model=model, save=False).dataset(dst)
        V = [v for v in V if (v is not None) and (not v.hasattribute('unrefined'))]  # remove videos that failed refinement
        V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
        return self.new(V, dst)

    def nondegenerate(self, src, dst=None):
        dst = dst if dst is not None else '%s_nondegenerate' % self.load(src).id()
        f = lambda v: v.clone() if ((v is not None) and (v.trackbox() is not None) and (not v.trackbox().intersection(v.clone().framebox(), strict=False).isdegenerate())) else None
        return self.map(src, f, dst=dst, ascompleted=False)

    def trackcrop(self, src, dst=None, dilate=1.0, maxsquare=True):
        dst = dst if dst is not None else '%s_trackcrop' % self.load(src).id()
        
        # FIXME: schema is broken
        f_saveas = lambda v, dst=dst, indir=self._indir: os.path.join(indir, dst, v.category(), filetail(v.filename()))
        f_trackcrop = lambda v, d=dilate, f=f_saveas, b=maxsquare: v.clone().trackcrop(dilate=d, maxsquare=b).saveas(f(v), flush=True).pack().print(sleep=1) if v is not None and v.clone().trackfilter(lambda t: len(t)>0).hastracks() else None
        return self.map(src, f_trackcrop, ascompleted=False).id(dst)

    def tubelet(self, src, dst, dilate=1.0, maxdim=512):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_tubelet = lambda v, d=dilate, m=maxdim, f=f_saveas: v.activitytube(dilate=d, maxdim=512).saveas(f(v)).print() if v.hastracks() else None
        return self.map(src, dst, f_tubelet)        

    def mindim(self, src, mindim, dstdir=None):
        dstdir = dstdir if dstdir is not None else 'mindim_%d' % mindim
        f_saveas = lambda v, dstdir=dstdir, f=self._schema: f(dstdir, v)
        f_tubelet = lambda v, m=mindim, f=f_saveas: v.mindim(m).saveas(f(v)).print()
        return self.map(src, f_tubelet).id('mindim_%d' % mindim)        

    def activitymontage(self, src, outfile, gridrows=30, gridcols=50, mindim=64, bycategory=False):
        """30x50 activity montage, each 64x64 elements using the output of prepare_dataset"""
        vidlist = self.dataset(src)
        actlist = [v.mindim(mindim) for v in vidlist]
        np.random.seed(42); random.shuffle(actlist)
        actlist = actlist[0:gridrows*gridcols]
        return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols).saveas(outfile).filename()

    def stabilizemontage(self, src, outfile, gridrows=6, gridcols=10):
         return vipy.visualize.videomontage([a.flush().crop(a.trackbox(dilate=1.2).maxsquare()).mindim(256).annotate() for a in self.take(src, gridrows*gridcols)], 128, 128, gridrows, gridcols).saveas(outfile)
         
    def activitymontage_bycategory(self, src, outfile, gridcols=49, mindim=64):
        """num_categoryes x gridcols activity montage, each row is a category"""
        np.random.seed(42)
        vidlist = self.dataset(src)
        categories = list(set([v.category() for v in vidlist]))

        actlist = []
        for k in sorted(categories):
            actlist_k = [v for v in vidlist if v.category() == k]
            random.shuffle(actlist_k)
            assert len(actlist_k) >= gridcols
            actlist.extend(actlist_k[0:gridcols])
            outfile = os.path.join(filepath(outfile), '%s_%s.mp4' % (filebase(outfile), k))
            print(vipy.visualize.videomontage(actlist_k[0:15], 256, 256, gridrows=3, gridcols=5).saveas(outfile).filename())

        outfile = os.path.join(filepath(outfile), '%s_%d_bycategory.mp4' % (filebase(outfile), gridcols))
        print('[pycollector.dataset.activitymontage_bycategory]: rows=%s' % str(sorted(categories)))
        return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=len(categories), gridcols=gridcols).saveas(outfile).filename()

    def copy(self, src, dst):
        return self.new(self.load(src).list(), dst)



class TorchDataset(torch.utils.data.Dataset):
    """Converter from a pycollector dataset to a torch dataset"""
    def __init__(self, f_transformer, d):
        assert isinstance(d, Dataset), "Invalid input"
        self._dataset = d
        self._f_transformer = f_transformer

    def __getitem__(self, k):
        return self._f_transformer(self._dataset[k])

    def __len__(self):
        return len(self._dataset)


class TorchDatadir(torch.utils.data.Dataset):
    """A torch dataset stored as a directory of .json files each containing a single object for training, and a transformer lambda function to conver this object to a tensor/label for data augmented training"""
    def __init__(self, f_transformer, jsondir):
        assert os.path.isdir(jsondir)
        self._jsonlist = vipy.util.listjson(jsondir)
        self._f_transformer = f_transformer

    def __getitem__(self, k):
        return self._f_transformer(vipy.util.load(self._jsonlist[k]))

    def __len__(self):
        return len(self._jsonlist)


class TorchTensordir(torch.utils.data.Dataset):
    """A torch dataset stored as a directory of .pkl.bz2 files each containing a list of tensor/labels used for data augmented training"""
    def __init__(self, tensordir):
        assert os.path.isdir(tensordir)
        self._dirlist = sorted([s for s in vipy.util.extlist(tensordir, '.pkl.bz2')], key=lambda d: int(vipy.util.filetail(d).rsplit('_', 1)[1].split('.pkl.bz2')[0]))

    def __getitem__(self, k):
        assert k >= 0 and k < len(self._dirlist)
        for j in range(0,2):
            try:
                obj = vipy.util.bz2pkl(self._dirlist[k])
                assert len(obj) > 0, "Invalid augmentation"
                return obj[np.random.randint(0, len(obj))]  # choose one tensor at random
            except:
                time.sleep(1)  # try again after a bit if another process is augmenting this .pkl.bz2 in parallel
        print('ERROR: %s corrupted' % self._dirlist[k])
        return self.__getitem__(np.random.randint(0, len(self)))  # maximum retries reached, get another one

    def __len__(self):
        return len(self._dirlist)

    
class Dataset():
    """pycollector.dataset.Dataset() class
    
       This class is designed to be used with vipy.batch.Batch() for massively parallel operations 
    """

    def __init__(self, objlist, id=None):
        assert isinstance(objlist, list), "Invalid input"
        self._saveas_ext = ['pkl', 'json']
        self._id = uuid.uuid4().hex if id is None else id
        self._objlist = tolist(objlist)
        assert len(self._objlist) > 0, "Invalid object list"

    def __repr__(self):
        return str('<pycollector.dataset: id="%s", len=%d, type=%s>' % (self.id(), len(self), str(type(self._objlist[0]))))

    def __iter__(self):
        for k in range(len(self)):
            yield self._objlist[k]

    def __getitem__(self, k):
        assert k>=0 and k<len(self._objlist), "invalid index"
        return self._objlist[k]

    def __len__(self):
        return len(self._objlist)

    def id(self, n=None):
        if n is None:
            return self._id
        else:
            self._id = n
            return self

    def list(self):
        return self._objlist

    def istype(self, validtype):
        return all([any([isinstance(v,t) for t in tolist(validtype)]) for v in self._objlist]), "invalid type - must be %s" % str(validtype)            
            
    def isvipy(self):
        return self.istype([vipy.image.Image, vipy.video.Video])

    def clone(self):
        return copy.deepcopy(self)

    def archive(self, tarfile, delprefix, mediadir='', format='json', castas=vipy.video.Scene, verbose=False, extrafiles=None):
        """Create a archive file for this dataset.  This will be archived as:

           tarfilename.{tar.gz|.tgz|.bz2}
              tarfilename
                 tarfilename.{json|pkl}
                 mediadir/
                     $SUBDIR/
                         video.mp4
                 extras1.ext
                 extras2.ext
        
            Inputs:
              - tarfile: /path/to/tarfilename.tar.gz
              - delprefix:  the absolute file path contained in the media filenames to be removed.  If a video has a delprefix='/a/b' then videos with path /a/b/c/d.mp4' -> 'c/d.mp4'
              - mediadir:  the subdirectory name of the media to be contained in the archive.  Usually "videos".             
              - extrafiles: list of tuples [(abspath, filename_in_archive),...]
        """
        assert self.isvipy(), "Source dataset must contain vipy objects for staging"
        assert all([os.path.isabs(v.filename()) for v in self]), "Input dataset must have only absolute media paths"
        assert self.countby(lambda v: delprefix in v.filename()) > 0, "delprefix not found"
        assert self.countby(lambda v: delprefix in v.filename()) == len(self), "all media objects must have the same delprefix for relative path construction"
        assert vipy.util.istgz(tarfile) or vipy.util.isbz2(tarfile), "Allowable extensions are .tar.gz, .tgz or .bz2"
        assert shutil.which('tar') is not None, "tar not found on path"        

        D = self.clone()
        stagedir = remkdir(os.path.join(tempdir(), filefull(filetail(tarfile))))
        print('[pycollector.dataset]: creating staging directory "%s"' % stagedir)        
        D._objlist = [v.filename(v.filename().replace(os.path.normpath(delprefix), os.path.normpath(os.path.join(stagedir, mediadir))), symlink=True) for v in D.list()]
        pklfile = os.path.join(stagedir, '%s.%s' % (filetail(filefull(tarfile)), format))
        D.save(pklfile, relpath=True, nourl=True, noadmin=True, castas=castas, significant_digits=2, noemail=True, flush=True)
    
        # Copy extras (symlinked) to staging directory
        if extrafiles is not None:
            for (e, a) in tolist(extrafiles):
                assert os.path.exists(e), "Invalid extras file '%s'" % e
                os.symlink(e, os.path.join(stagedir, filetail(e) if a is None else a))

        # System command to run tar
        cmd = ('tar %scvf %s -C %s --dereference %s %s' % ('j' if vipy.util.isbz2(tarfile) else 'z', 
                                                           tarfile,
                                                           filepath(stagedir),
                                                           filetail(stagedir),
                                                           ' > /dev/null' if not verbose else ''))

        print('[pycollector.dataset]: executing "%s"' % cmd)        
        os.system(cmd)  # too slow to use python "tarfile" package
        print('[pycollector.dataset]: deleting staging directory "%s"' % stagedir)        
        shutil.rmtree(stagedir)
        print('[pycollector.dataset]: %s, MD5=%s' % (tarfile, vipy.downloader.generate_md5(tarfile)))
        return tarfile
        
    def save(self, outfile, nourl=False, castas=None, relpath=False, noadmin=False, strict=True, significant_digits=2, noemail=True, flush=True):    
        n = len([v for v in self._objlist if v is None])
        if n > 0:
            print('[pycollector.dataset]: removing %d invalid elements' % n)
        objlist = [v for v in self._objlist if v is not None]  
        if relpath or nourl or noadmin or flush or noemail or (significant_digits is not None):
            assert self.isvipy(), "Invalid input"
        if relpath:
            print('[pycollector.dataset]: setting relative paths')
            objlist = [v.relpath(filepath(outfile)) if os.path.isabs(v.filename()) else v for v in objlist]
        if nourl: 
            print('[pycollector.dataset]: removing URLs')
            objlist = [v.nourl() for v in objlist]           
        if noadmin:
            objlist = [v.delattribute('admin') for v in objlist]
        if castas is not None:
            assert hasattr(castas, 'cast'), "Invalid cast"
            print('[pycollector.dataset]: casting as "%s"' % (str(castas)))
            objlist = [castas.cast(v) for v in objlist]                     
        if significant_digits is not None:
            assert isinstance(significant_digits, int) and significant_digits >= 1, "Invalid input"
            objlist = [o.trackmap(lambda t: t.significant_digits(significant_digits)) if o is not None else o for o in objlist]
        if noemail:
            for o in objlist:
                for (k,v) in o.attributes.items():
                    if isinstance(v, str) and is_email_address(v):
                        o.attributes[k] = hashlib.sha1(v.encode("UTF-8")).hexdigest()[0:10]
        if flush:
            objlist = [o.flush() for o in objlist]  

        print('[pycollector.dataset]: Saving %s to "%s"' % (str(self), outfile))
        vipy.util.save(objlist, outfile)
        return self

    def classlist(self):
        assert self.isvipy(), "Invalid input"
        return sorted(list(set([v.category() for v in self._objlist])))

    def class_to_index(self):
        return {v:k for (k,v) in enumerate(self.classlist())}

    def index_to_class(self):
        return {v:k for (k,v) in self.class_to_index().items()}

    def label_to_index(self):
        return self.class_to_index()

    def powerset(self):
        return list(sorted(set([tuple(sorted(list(a))) for v in self._objlist for a in v.activitylabel() if len(a) > 0])))        

    def powerset_to_index(self):        
        assert self.isvipy(), "Invalid input"
        return {c:k for (k,c) in enumerate(self.powerset())}

    def dedupe(self, key):
        self._objlist = list({key(v):v for v in self._objlist}.values())
        return self
        
    def countby(self, f):
        return len([v for v in self._objlist if f(v)])

    def union(self, other, key=None):
        assert isinstance(other, Dataset), "invalid input"
        self._objlist = self._objlist + other._objlist
        return self.dedupe(key) if key is not None else self
    
    def difference(self, other, key):
        assert isinstance(other, Dataset), "invalid input"
        idset = set([key(v) for v in self._objlist]).difference([key(v) for v in other._objlist])   # in A but not in B
        self._objlist = [v for v in self._objlist if key(v) in idset]
        return self
        
    def filter(self, f):
        self._objlist = [v for v in self._objlist if f(v)]
        return self

    def take_per_category(self, n, id=None):
        C = vipy.util.groupbyasdict(self._objlist, lambda v: v.category())
        return Dataset([C[k][j] for (k,v) in C.items() for j in np.random.permutation(list(range(len(v))))[0:n]], id=id)

    def tojsondir(self, outdir):
        print('[pycollector.dataset]: exporting %d json files to "%s"...' % (len(self), outdir))
        vipy.util.remkdir(outdir)  # to avoid race condition
        Batch(vipy.util.chunklist([(k,v) for (k,v) in enumerate(self._objlist)], 64), as_completed=True, minscatter=1).map(lambda X: [vipy.util.save(x[1].clone(), os.path.join(outdir, '%s_%d.json' % (x[1].clone().videoid(), x[0]))) for x in X]).result()
        return outdir

    def takelist(self, n, category=None):
        objlist = self._objlist if category is None else [v for v in self._objlist if v.category() == category]
        assert n <= len(objlist), "Invalid length"
        return [objlist[k] for k in random.sample(range(0, len(objlist)), n)]  # without replacement

    def take(self, n, category=None):
        return Dataset(self.takelist(n, category))
    
    def split(self, trainfraction=0.9, valfraction=0.1, testfraction=0, seed=42):
        """Split the dataset by category by fraction so that video IDs are never in the same set"""
        assert self.isvipy(), "Invalid input"
        assert trainfraction >=0 and trainfraction <= 1
        assert valfraction >=0 and valfraction <= 1
        assert testfraction >=0 and testfraction <= 1
        assert trainfraction + valfraction + testfraction == 1.0

        np.random.seed(seed)
        A = self.list()
        
        # Video ID assignment
        videoid = list(set([a.videoid() for a in A]))
        np.random.shuffle(videoid)
        (testid, valid, trainid) = vipy.util.dividelist(videoid, (testfraction, valfraction, trainfraction))        
        (testid, valid, trainid) = (set(testid), set(valid), set(trainid))
        d = vipy.util.groupbyasdict(A, lambda a: 'testset' if a.videoid() in testid else 'valset' if a.videoid() in valid else 'trainset')
        (trainset, testset, valset) = (d['trainset'] if 'trainset' in d else [], 
                                       d['testset'] if 'testset' in d else [], 
                                       d['valset'] if 'valset' in d else [])

        print('[pycollector.dataset]: trainset=%d (%1.1f)' % (len(trainset), trainfraction))
        print('[pycollector.dataset]: valset=%d (%1.1f)' % (len(valset), valfraction))
        print('[pycollector.dataset]: testset=%d (%1.1f)' % (len(testset), testfraction))
        
        return (Dataset(trainset, id='trainset'), Dataset(valset, id='valset'), Dataset(testset, id='testset') if len(testset)>0 else None)

    def tocsv(self, csvfile=None):
        csv = [v.csv() for v in self.list]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])

    def map(self, f_transform, model=None, dst=None, checkpoint=False, strict=False, ascompleted=True):        
        B = Batch(self.list(), strict=strict, as_completed=ascompleted, checkpoint=checkpoint, warnme=False, minscatter=1000000)
        V = B.map(f_transform).result() if not model else B.scattermap(f_transform, model).result() 
        if any([v is None for v in V]):
            print('pycollector.datasets][%s->]: %d failed' % (str(self), len([v for v in V if v is None])))
        return Dataset(V, id=dst)

    def localmap(self, f):
        self._objlist = [f(v) for v in self._objlist]
        return self

    def flatmap(self, f):
        self._objlist = [x for v in self._objlist for x in f(v)]
        return self
    
    def count(self):
        """Counts for each label"""
        assert self.isvipy()
        return vipy.util.countby(self.list(), lambda v: v.category())

    def percentage(self):
        """Fraction of dataset for each label"""
        d = self.count()
        n = sum(d.values())
        return {k:v/float(n) for (k,v) in d.items()}

    def multilabel_inverse_frequency_weight(self):
        fw = {k:0 for k in self.classlist()}
        for v in self.list():
            lbl_frequency = vipy.util.countby([a for A in v.activitylabel() for a in A], lambda x: x)  # frequency within clip
            lbl_weight = {k:(f/float(max(1, sum(lbl_frequency.values())))) for (k,f) in lbl_frequency.items()}  # multi-label likelihood within clip, sums to one
            for (k,w) in lbl_weight.items():
                fw[k] += w

        n = sum(fw.values())  
        ifw = {k:1.0/((w/n)*len(fw)) for (k,w) in fw.items()}
        return ifw

    def inverse_frequency_weight(self):
        d = {k:1.0/max(v,1) for (k,v) in self.count().items()}
        n = sum(d.values())
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def collectors(self, outfile=None):
        assert self.isvipy()
        d = vipy.util.countby(self.list(), lambda v: v.attributes['collector_id'])
        f = lambda x,n: len([k for (k,v) in d.items() if int(v) >= n])
        print('[collector.dataset.collectors]: Collectors = %d ' % f(d,0))
        print('[collector.dataset.collectors]: Collectors with >10 submissions = %d' % f(d,10))
        print('[collector.dataset.collectors]: Collectors with >100 submissions = %d' % f(d,100))
        print('[collector.dataset.collectors]: Collectors with >1000 submissions = %d' % f(d,1000))
        print('[collector.dataset.collectors]: Collectors with >10000 submissions = %d' % f(d,10000))

        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), list(range(len(d.keys()))), outfile=outfile, ylabel='Submissions', xlabel='Collector ID', xrot='vertical', fontsize=3, xshow=False)            
        return d

    def os(self, outfile=None):
        assert self.isvipy()
        d = vipy.util.countby([v for v in self.list() if v.hasattribute('device_identifier')], lambda v: v.attributes['device_identifier'])
        print('[collector.dataset.collectors]: Device OS = %d ' % len(d))
        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), d.keys(), explode=None, outfile=outfile,  shadow=False)
        return d

    def device(self, outfile=None, n=24, fontsize=7):
        d_all = vipy.util.countby([v for v in self.list() if v.hasattribute('device_type') and v.attributes['device_type'] != 'unrecognized'], lambda v: v.attributes['device_type'])
        
        topk = [k for (k,v) in sorted(list(d_all.items()), key=lambda x: x[1])[-n:]] 
        other = np.sum([v for (k,v) in d_all.items() if k not in set(topk)])

        d = {k:v for (k,v) in d_all.items() if k in set(topk)}
        d.update( {'Other':other} )
        d = dict(sorted(list(d.items()), key=lambda x: x[1]))

        print('[collector.dataset.collectors]: Device types = %d ' % len(d_all))
        print('[collector.dataset.collectors]: Top-%d Device types = %s ' % (n, str(topk)))

        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), d.keys(), explode=None, outfile=outfile,  shadow=False, legend=False, fontsize=fontsize, rotatelabels=False)
        return d
        
    def duration_in_frames(self, outfile=None):
        assert self.isvipy()
        d = {k:np.mean([v[1] for v in v]) for (k,v) in vipy.util.groupbyasdict([(a.category(), len(a)) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (frames)', fontsize=6)            
        return d

    def duration_in_seconds(self, outfile=None):
        assert self.isvipy()
        d = {k:np.mean([v[1] for v in v]) for (k,v) in vipy.util.groupbyasdict([(a.category(), len(a)/v.framerate()) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (seconds)', fontsize=6)            
        return d

    def framerate(self, outfile=None):
        assert self.isvipy()
        d = vipy.util.countby([int(round(v.framerate())) for v in self.list()], lambda x: x)
        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), ['%d fps' % k for k in d.keys()], explode=None, outfile=outfile,  shadow=False)
        return d
        
        
    def density(self, outfile=None):
        assert self.isvipy()
        d = [len(v) for (k,v) in vipy.util.groupbyasdict(self.list(), lambda v: v.videoid()).items()]
        d = vipy.util.countby(d, lambda x: x)
        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Frequency', xlabel='Activities per video', fontsize=6, xrot=None)            
        return d
        

    def stats(self, outdir=None, object_categories=['Person', 'Car'], plot=True):
        """Analyze the dataset to return helpful statistics and plots"""
        assert self.isvipy()

        videos = self.list()
        #scenes = [a for m in videos for a in m.activityclip() if m is not None]  # This can introduce doubles
        scenes = videos
        activities = [a for s in scenes for a in s.activities().values()]
        tracks = [t for s in scenes for t in s.tracks().values()]
        outdir = tempdir() if outdir is None else outdir
        
        # Category distributions
        d = {}
        d['activity_categories'] = set([a.category() for a in activities])
        d['object_categories'] = set([t.category() for t in tracks])
        #d['videos'] = set([v.filename() for v in videos if v is not None])
        d['num_activities'] = sorted([(k,len(v)) for (k,v) in vipy.util.groupbyasdict(activities, lambda a: a.category()).items()], key=lambda x: x[1])
        #d['video_density'] = sorted([(v.filename(),len(v.activities())) for v in videos if v is not None], key=lambda x: x[1])

        # Helpful plots
        if plot:
            import matplotlib.pyplot as plt        
            import vipy.metrics
            from vipy.show import colorlist        
            
            # Histogram of instances
            (categories, freq) = zip(*reversed(d['num_activities']))
            barcolors = ['blue' if c.startswith('person') else 'green' for c in categories]
            d['num_activities_histogram'] = vipy.metrics.histogram(freq, categories, barcolors=barcolors, outfile=os.path.join(outdir, 'num_activities_histogram.pdf'), ylabel='Instances', fontsize=6)
            colors = colorlist()

            # Scatterplot of people and vehicles box sizes
            (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
            plt.clf()
            plt.figure()
            plt.grid(True)
            d_category_to_color = dict(zip(object_categories, ['blue', 'green']))
            for c in object_categories:
                xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category() == c) and (t.meanshape() is not None))]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 1000, 0, 1000])                
            plt.legend()
            plt.gca().set_axisbelow(True)        
            d['object_bounding_box_scatterplot'] = os.path.join(outdir, 'object_bounding_box_scatterplot.pdf')
            plt.savefig(d['object_bounding_box_scatterplot'])
        
            # 2D histogram of people and vehicles box sizes
            for c in object_categories:
                xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category() == c) and (t.meanshape() is not None))]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.clf()
                    plt.figure()
                    plt.hist2d(xc, yc, bins=10)
                    plt.xlabel('Bounding box (width)')
                    plt.ylabel('Bounding box (height)')
                    
                    d['2D_%s_bounding_box_histogram' % c] = os.path.join(outdir, '2D_%s_bounding_box_histogram.pdf' % c)
                    plt.savefig(d['2D_%s_bounding_box_histogram' % c])

            # Mean track size per activity category
            d_category_to_xy = {k:np.mean([t.meanshape() for v in vlist for t in v.tracklist()], axis=0) for (k,vlist) in vipy.util.groupbyasdict(scenes, lambda v: v.category()).items()}        
            plt.clf()
            plt.figure()
            plt.grid(True)
            d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(d_category_to_xy.keys())}
            for c in d_category_to_xy.keys():
                (xc, yc) = d_category_to_xy[c]
                plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 600, 0, 600])                
            plt.gca().set_axisbelow(True)        
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            d['activity_bounding_box_scatterplot'] = os.path.join(outdir, 'activity_bounding_box_scatterplot.pdf')
            plt.savefig(d['activity_bounding_box_scatterplot'], bbox_extra_artists=(lgd,), bbox_inches='tight')
    
        return d

    def totorch(self, f_video_to_tensor):
        return TorchDataset(f_video_to_tensor, self)

    def totorch_datadir(self, f_video_to_tensor, outdir):
        return TorchDatadir(f_video_to_tensor, self.tojsondir(outdir))

    def tohtml(self, outfile, mindim=512, title='Visualization', fraction=1.0, display=False, clip=True, activities=True, category=True):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
        assert ishtml(outfile), "Output file must be .html"
        assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"
        
        import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
        import vipy.batch  # requires pip install vipy[all]

        dataset = self.list()
        assert all([isinstance(v, vipy.video.Video) for v in dataset])
        dataset = [dataset[k] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
        dataset = [v for v in dataset if all([len(a) < 15*v.framerate() for a in v.activitylist()])]  # remove extremely long videos

        quicklist = vipy.batch.Batch(dataset, strict=False, as_completed=True, minscatter=1).map(lambda v: (v.load().quicklook(), v.flush().print())).result()
        quicklist = [x for x in quicklist if x is not None]  # remove errors
        quicklooks = [imq for (imq, v) in quicklist]  # keep original video for HTML display purposes
        provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)


    
class ActivityDataset(Dataset):
    """Dataset filename structure: /dataset/$DSTDIR/$ACTIVITY_CATEGORY/$VIDEOID_$ACTIVITYINDEX.mp4"""
    def __init__(self, indir, format='.json'):
        super().__init__(indir, format)

class FaceDataset(Dataset):
    """Dataset filename structure: /outdir/$SUBDIR/$SUBJECT_ID/$VIDEOID.mp4""" 
    def __init__(self, indir, format='.json'):
        super().__init__(indir, format)
        self._schema = lambda dstdir, v, indir=self._indir, schema=self._schema: os.path.join(indir, dstdir, v.subjectid(), filetail(v.filename()))    
        raise

    def load(self, transform):
        V = super().load(transform)
        assert all([isinstance(v, pycollector.video.Video) for v in V]), "Face dataset requires subject ID"
        return V
    
class ObjectDataset(Dataset):
    """Dataset filename structure: /outdir/$SUBDIR/$OBJECT_CATEGORY/$VIDEOID.mp4"""     
    def __init__(self, indir, format='.json'):
        super().__init__(indir, format)
        self._schema = lambda dstdir, v, indir=self._indir, schema=self._schema: os.path.join(indir, dstdir, v.category(), filetail(v.filename()))    
        raise

