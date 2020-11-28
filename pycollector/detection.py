import os
import sys
import torch
import vipy
import shutil
import numpy as np
import warnings
from vipy.util import remkdir, filetail, readlist, tolist, filepath, chunklistbysize, Timer
from pycollector.video import Video
from pycollector.model.yolov3.network import Darknet
from pycollector.globals import print
from pycollector.model.face.detection import FaceRCNN 
import copy


class TorchNet(object):

    def gpu(self, idlist, batchsize=None):
        assert batchsize is None or (isinstance(batchsize, int) and batchsize > 0), "Batchsize must be integer"
        assert idlist is None or isinstance(idlist, int) or (isinstance(idlist, list) and len(idlist)>0), "Input must be a non-empty list of integer GPU ids"
        self._batchsize = batchsize if batchsize is not None else (self._batchsize if hasattr(self, '_batchsize') else 1)

        idlist = tolist(idlist)
        self._devices = ['cuda:%d' % k if k is not None and torch.cuda.is_available() else 'cpu' for k in idlist]
        #self._tensortype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor       
        self._tensortype = torch.FloatTensor       
        
        if not hasattr(self, '_gpulist') or not hasattr(self, '_models') or idlist != self._gpulist:
            self._models = [copy.deepcopy(self._model).to(d, non_blocking=False) for d in self._devices]
            for m in self._models:
                m.eval()
            self._gpulist = idlist

        return self

    def __call__(self, t):
        """Parallel evaluation of tensor to split across GPUs set up in gpu().  t should be of size (ngpu*batchsize)
        
           * Note: Do not use DataParallel, this replicates the multi-gpu batch on device 0 and results in out of memory
        """
        assert len(t) <= self.batchsize(), "Invalid batch size"
        todevice = [b.to(d, non_blocking=True) for (b,d) in zip(t.pin_memory().split(self._batchsize), self._devices)]  # async?
        fromdevice = [m(b) for (m,b) in zip(self._models, todevice)]   # async?
        return torch.cat([r.detach().cpu() for r in fromdevice], dim=0)
        
    def batchsize(self):
        return len(self._models)*self._batchsize
        

class FaceDetector(TorchNet):
    """Faster R-CNN based face detector
    
    """

    def __init__(self, weightfile=None):    
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'face')

        weightfile = os.path.join(indir, 'resnet-101_faster_rcnn_ohem_iter_20000.pth') if weightfile is None else weightfile
        if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, 'a759030540a4a5284baa93d3ef5e47ed40cae6d6'):
            print('[pycollector.detection]: Downloading face detector weights ...')
            os.system('wget -c https://dl.dropboxusercontent.com/s/rdfre0oc456t5ee/resnet-101_faster_rcnn_ohem_iter_20000.pth -O %s' % weightfile)  # FIXME: replace with better solution
        assert vipy.downloader.verify_sha1(weightfile, 'a759030540a4a5284baa93d3ef5e47ed40cae6d6'), "Face detector download failed with SHA1='%s'" % (vipy.downloader.generate_sha1(weightfile))
        self._model = FaceRCNN(model_path=weightfile)
        #self._model.eval()  # Set in evaluation mode
        #self.gpu(vipy.globals.gpuindex())

    def __call__(self, im):
        assert isinstance(im, vipy.image.Image)
        return vipy.image.Scene(array=im.numpy(), colorspace=im.colorspace(), objects=[vipy.object.Detection('face', xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3], confidence=bb[4]) for bb in self._model(im)]).union(im)


class ObjectDetector(TorchNet):
    """Yolov3 based object detector

       >>> d = pycollector.detection.Detector()
       >>> d(vipy.image.vehicles()).show()

    """
    
    def __init__(self, batchsize=1, weightfile=None, gpu=None):    
        self._mindim = 416  # must be square
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'yolov3')
        weightfile = os.path.join(indir, 'yolov3.weights') if weightfile is None else weightfile
        cfgfile = os.path.join(indir, 'yolov3.cfg')
        self._model = Darknet(cfgfile, img_size=self._mindim)
        if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, '520878f12e97cf820529daea502acca380f1cb8e'):
            #vipy.downloader.download('https://www.dropbox.com/s/ve9cpuozbxh601r/yolov3.weights', os.path.join(indir, 'yolov3.weights'))
            print('[pycollector.detection]: Downloading object detector weights ...')
            os.system('wget -c https://dl.dropboxusercontent.com/s/ve9cpuozbxh601r/yolov3.weights -O %s' % weightfile)  # FIXME: replace with better solution
        assert vipy.downloader.verify_sha1(weightfile, '520878f12e97cf820529daea502acca380f1cb8e'), "Object detector download failed"
        self._model.load_darknet_weights(weightfile)
        self._model.eval()  # Set in evaluation mode
        self._batchsize = batchsize        
        assert isinstance(self._batchsize, int), "Batchsize must be integer"
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self._index2cls = {k:c for (c,k) in self._cls2index.items()}

        self._device = None
        self._gpulist = gpu
        self.gpu(gpu, batchsize)
        
    def __call__(self, im, conf=5E-1, iou=0.5, union=False, objects=None):
        assert isinstance(im, vipy.image.Image) or (isinstance(im, list) and all([isinstance(i, vipy.image.Image) for i in im])), "Invalid input - must be vipy.image.Image object and not '%s'" % (str(type(im)))

        imlist = tolist(im)
        imlistdets = []
        t = torch.cat([im.clone().maxsquare().mindim(self._mindim).mat2gray().torch() for im in imlist]).type(self._tensortype)  # triggers load
        t_out = super().__call__(t).numpy()   # parallel multi-GPU evaluation, using TorchNet()
        for (im, dets) in zip(imlist, t_out):
            k_class = np.argmax(dets[:,5:], axis=1).flatten().tolist()
            k_det = np.argwhere((dets[:,4] > conf).flatten() & np.array([((objects is None) or (self._index2cls[k] in set(objects))) for k in k_class])).flatten().tolist()
            objectlist = [vipy.object.Detection(xcentroid=float(dets[k][0]),
                                                ycentroid=float(dets[k][1]),
                                                width=float(dets[k][2]),
                                                height=float(dets[k][3]),
                                                confidence=float(dets[k][4]),
                                                category='%s' % self._index2cls[k_class[k]],
                                                id=True)
                          for k in k_det]
            
            scale = max(im.shape()) / float(self._mindim)  # to undo
            objectlist = [obj.rescale(scale) for obj in objectlist]
            imd = im.clone().array(im.numpy()).objects(objectlist).nms(conf, iou)  # clone for shared attributese
            imlistdets.append(imd if not union else imd.union(im))
            
        return imlistdets if self._batchsize > 1 else imlistdets[0]

    def classlist(self):
        return list(self._cls2index.keys())
    

class MultiscaleObjectDetector(ObjectDetector):
    """Run VideoDetector on every frame of video, tiling the frame to an overlapping  mosaic each of size (self._mindim, self._mindim), then recombining detections"""
    def __call__(self, imlist, conf=0.5, iou=0.5, maxarea=1.0, objects=None):
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(imlist, vipy.image.Image) or isinstance(imlist, list) and all([isinstance(im, vipy.image.Image) for im in imlist]), "invalid input"
        imlist = tolist(imlist)

        (imlist_multiscale, imlist_multiscale_flat, n_coarse, n_fine) = ([], [], [], [])
        for im in imlist:
            imcoarse = im.clone().mindim(n).tile(n, n, overlaprows=n//2, overlapcols=n//2)
            imfine = im.clone().tile(n, n, overlaprows=n//2, overlapcols=n//2) if im.mindim() > (n+(n//2)) else []
            n_coarse.append(len(imcoarse))
            n_fine.append(len(imfine))
            imlist_multiscale.append(imcoarse+imfine)
            imlist_multiscale_flat.extend(imcoarse+imfine)

        imlist_multiscale_flat = [im.maxsquare().cornerpadcrop(n,n) for im in imlist_multiscale_flat]
        imlistdet_multiscale_flat = [im for iml in chunklistbysize(imlist_multiscale_flat, self.batchsize()) for im in f(iml, objects=objects)]
        imlistdet = []
        for (k, (iml, imb, nf, nc)) in enumerate(zip(imlist, imlist_multiscale, n_fine, n_coarse)):
            im_multiscale = imlistdet_multiscale_flat[0:nf+nc]; imlistdet_multiscale_flat = imlistdet_multiscale_flat[nf+nc:];
            imcoarsedet = iml.clone().untile(im_multiscale[0:nc]).mindim(iml.mindim()).nms(conf, iou)
            imfinedet = iml.clone().untile( [im.objectfilter(lambda o: o.area()<=maxarea*im.area() and o.clone().dilate(1.1).isinterior(im.width(), im.height()))  # not too big or occluded by image boundary 
                                             for im in im_multiscale[nc:]] )
            imcoarsedet = imcoarsedet.union(imfinedet, iou) if imfinedet is not None else imcoarsedet            
            imlistdet.append(imcoarsedet.nms(conf, iou))

        return imlistdet[0] if len(imlistdet) == 1 else imlistdet

    
class VideoDetector(ObjectDetector):  
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"        
        for im in v.stream():
            yield super().__call__(im, conf=conf, iou=iou)

            
class MultiscaleVideoDetector(MultiscaleObjectDetector):
    """Run VideoDetector on every frame of video, tiling the frame to a mosaic each of size (self._mindim, self._mindim)"""
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"
        for imf in v.stream():
            yield super().__call__(imf, conf, iou)


class VideoTracker(ObjectDetector):
    def __call__(self, v, conf=0.05, iou=0.5, maxhistory=5, smoothing=None, objects=None, mincover=0.8, maxconf=0.2):
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(v, vipy.video.Video), "Invalid input"
        assert objects is None or all([o in self.classlist() for o in objects]), "Invalid object list"
        vc = v.clone().clear()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate(f(vb.framelist(), conf, iou, union=False, objects=objects)):
                yield vc.assign(k*self.batchsize()+j, im.clone().objectfilter(lambda o: o.category() in objects if objects is not None else True).objects(), miniou=iou, maxhistory=maxhistory, minconf=conf, maxconf=maxconf, mincover=mincover)  # in-place            

    def track(self, v, conf=0.05, iou=0.5, maxhistory=5, smoothing=None, objects=None, mincover=0.8, maxconf=0.2, verbose=False):
        for (k,vt) in enumerate(self.__call__(v.clone(), conf, iou, maxhistory, smoothing, objects, mincover, maxconf)):
            if verbose:
                print('[pycollector.detection.VideoTracker][%d]: %s' % (k, str(vt)))  
        return vt

    

class MultiscaleVideoTracker(MultiscaleObjectDetector):
    def __call__(self, v, conf=0.05, iou=0.6, maxarea=1.0, maxhistory=5, smoothing=None, objects=None, mincover=0.6, maxconf=0.2):
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(v, vipy.video.Video), "Invalid input"
        assert objects is None or all([o in self.classlist() for o in objects]), "Invalid object list"
        vc = v.clone().clear()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate(f(vb.framelist(), conf, iou, maxarea, objects=objects)):
                yield vc.assign(k*self.batchsize()+j, im.clone().objectfilter(lambda o: o.category() in objects if objects is not None else True).objects(), miniou=iou, maxhistory=maxhistory, minconf=conf, maxconf=maxconf, mincover=mincover)  # in-place            
        
    def track(self, v, conf=0.05, iou=0.6, maxhistory=5, smoothing=None, objects=None, mincover=0.6, maxconf=0.2, verbose=False):
        for (k,vt) in enumerate(self.__call__(v.clone(), conf, iou, 1.0, maxhistory, smoothing, objects, mincover, maxconf)):
            if verbose:
                print('[pycollector.detection.MultiscaleVideoTracker][%d]: %s' % (k, str(vt)))  
        return vt
        

class Proposal(ObjectDetector):
    def __call__(self, v, conf=1E-2, iou=0.8):
        return super().__call__(v, conf, iou)
    
        
class VideoProposal(Proposal):
    """pycollector.detection.VideoProposal() class.
    
       Track-based object proposals in video.
    """
    def allowable_objects(self):
        return ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle']

    def isallowable(self, v):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be vipy.video.Video not '%s'" % (str(type(v)))
        return len(set(v.objectlabels())) == 1 and all([c.lower() in self.allowable_objects() for c in v.objectlabels()]) # for now

    def __call__(self, v, conf=1E-2, iou=0.8, dt=1, target=None, activitybox=False, dilate=4.0, dilate_height=None, dilate_width=None):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be vipy.video.Video not '%s'" % (str(type(v)))
        self.gpu(vipy.globals.gpuindex())  # cpu if gpuindex == None

        # Optional target class
        d_target_to_index = {'person':[self._cls2index['person']],
                             'bicycle':[self._cls2index['bicycle']],
                             'vehicle':[self._cls2index['car'], self._cls2index['motorbike'], self._cls2index['truck']], 
                             'car':[self._cls2index['car'], self._cls2index['truck']],
                             'motorcycle':[self._cls2index['motorbike']],
                             'object':[self._cls2index[k] for k in ['backpack', 'handbag', 'suitcase', 'frisbee', 'sports ball', 'bottle', 'cup', 'bowl', 'laptop', 'book']]}
        d_index_to_target = {i:k for (k,v) in d_target_to_index.items() for i in v}
        assert all([k in self.allowable_objects() for k in d_target_to_index.keys()])        
        assert target is None or (isinstance(target, list) and all([t in d_target_to_index.keys() for t in target]))
        f_max_target_confidence = lambda d: (max([d[5+k] for t in target for k in d_target_to_index[t]]) if target is not None else max(d[5:]))
        f_max_target_category = lambda d: (sorted([(d[5+k], t) for t in target for k in d_target_to_index[t]], key=lambda x: x[0])[-1][1] if target is not None else None)
        
        # Source video foveation: dilated crop of the activity box and resize (this transformation must be reversed)
        vc = v.clone(flushforward=True)  # to avoid memory leaks
        (dilate_height, dilate_width) = (dilate if dilate_height is None else dilate_height, dilate if dilate_width is None else dilate_width)
        bb = vc.activitybox().dilate_height(dilate_height).dilate_width(dilate_width).imclipshape(vc.width(), vc.height()) if activitybox else vipy.geometry.imagebox(vc.shape())
        scale = max(bb.shape()) / float(self._mindim)  # for reversal, input is maxsquare() resized to _mindim

        # Batched proposals on transformed video (preloads source and transformed videos, high mem requirement)
        ims = []
        img = vc.numpy()[::dt]  # source video, triggers load
        tensor = vc.flush().crop(bb, zeropad=False).maxsquare().mindim(self._mindim).torch()[::dt]  # transformed video, NxCxHxW, re-triggers load due to crop()

        for i in range(0, len(tensor), self._batchsize):
            dets = self._model(tensor[i:i+self._batchsize].type(self._tensortype).to(self._device))  # copy here
            for (j, det) in enumerate(dets):
                # Objects in transformed video
                objs = [vipy.object.Detection(xcentroid=float(d[0]), 
                                              ycentroid=float(d[1]), 
                                              width=float(d[2]), 
                                              height=float(d[3]), 
                                              confidence=(float(d[4]) + f_max_target_confidence(d)),
                                              category=('%1.2f' % float(d[4])) if target is None else f_max_target_category(d))
                        for d in det
                        if (float(d[4]) > conf) and (f_max_target_confidence(d) > conf)]

                # Objects in source video
                objs = [obj.rescale(scale).translate(bb.xmin(), bb.ymin()) for obj in objs]
                ims.append( vipy.image.Scene(array=img[i+j], objects=objs).nms(conf, iou) )
                
        return ims

    
class VideoProposalRefinement(VideoProposal):
    """pycollector.detection.VideoProposalRefinement() class.
    
       Track-based object proposal refinement of a weakly supervised loose object box from a human annotator.
    """
    
    def __call__(self, v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=1, meanfilter=15, mincover=0.8, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True, byclass=True, dilate_height=None, dilate_width=None, refinedclass=None, pdist=False, minconf=1E-2):
        """Replace proposal in v by best (maximum overlap and confidence) object proposal in vc.  If no proposal exists, delete the proposal."""
        assert isinstance(v, vipy.video.Scene), "Invalid input - must be vipy.video.Scene not '%s'" % (str(type(v)))
        assert not (byclass is False and refinedclass is not None), "Invalid input"
        
        if not self.isallowable(v):
            warnings.warn("Invalid object labels '%s' for proposal, must be in '%s' - returning original video" % (str(v.objectlabels()), str(self.allowable_objects())))
            return v.clone().setattribute('unrefined')
        target = None if not byclass else [c.lower() for c in v.objectlabels()] if refinedclass is None else [refinedclass.lower()]  # classes for proposals
        vp = super().__call__(v, proposalconf, proposaliou, dt=dt, activitybox=True, dilate_height=dilate_height, dilate_width=dilate_width, target=target)  # subsampled proposals
        vc = v.clone(rekey=True, flushforward=True, flushbackward=True).trackfilter(lambda t: len(t) > dt)
        for (ti, t) in vc.tracks().items():
            if len(t) <= 1:
                continue  # no human annotation, skip

            t.resample(dt=dt)  # interpolated keyframes for source proposal
            bbprev = None  # last box assignment
            s = shapeiou  # shape assignment threshold
            for (f, bb) in zip(t.clone().keyframes(), t.clone().keyboxes()):  # clone because vc is being modified in-loop
                fs = int(f // dt)  # subsampled frame index, guaranteed incremental [0,1,2,...] by resample()
                if fs>=0 and fs<=len(vp):  
                    # Assignment: maximum (overlap with previous box + same shape as previous box + overlap with human box) * (objectness confidence + class confidence)
                    # Assignment constraints: (new box must not be too small relative to collector box) and (new box must be mostly contained within the collector box) and (new box must mostly overlap previous box) 
                    assignment = sorted([(bbp, (((bbp.shapeiou(bbprev) + bbp.iou(bbprev)) if bbprev is not None else 1.0) + (bb.iou(bbp) if not pdist else bb.pdist(bbp)))*bbp.confidence())
                                         for bbp in vp[min(fs, len(vp)-1)].objects() 
                                         if (bb.iou(bbp)>=miniou and   # refinement overlaps proposal (proposal is loose)
                                             bbp.cover(bb)>=mincover and  # refinement is covered by proposal (proposal is loose, refinement is inside)
                                             (bbprev is None or bbprev.shapeiou(bbp)>s) and  # refinement has similar shape over time
                                             (byclass is False or (refinedclass is None and (bbp.category().lower() == bb.category().lower())) or (refinedclass is not None and bbp.category().lower()==refinedclass.lower()))  # refine by target object only
                                         )], key=lambda x: x[1])

                    (bbp, iou) = assignment[-1] if len(assignment)>0 else (None, None)  # best assignment                    
                    if iou is not None and iou > minconf:
                        newcategory = t.category() if refinedclass is None else refinedclass
                        vc.tracks()[ti].category(newcategory).replace(f, bbp.clone().category(newcategory))
                        bbprev = bbp.clone() # update last assignment
                        s = shapeiou
                    else:
                        if strict:
                            vc.tracks()[ti].delete(f)  # Delete proposal that has no object proposal, otherwise use source proposal for interpolation
                        s = max(0, s-(0.01*dt))  # gate increase (shape assignment threshold decrease) for shape deformation

        # Remove empty tracks:
        # if a track does not have an assignment for the last (or first) source proposal, then it will be truncated here
        vc = vc.trackfilter(lambda t: len(t)>dt)    # may be empty

        # Proposal smoothing
        if not vc.hastracks():
            warnings.warn('VideoProposalRefinement returned no tracks')
            return vc.setattribute('unrefined')
        elif smoothing == 'mean':
            # Mean track smoothing: mean shape smoothing with mean coordinate smoothing with very small support for unstabilized video
            return vc.trackmap(lambda t: t.smoothshape(width=meanfilter//dt).smooth(3))
        elif smoothing == 'spline':
            # Cubic spline track smoothing with mean shape smoothing 
            return vc.trackmap(lambda t: t.smoothshape(width=meanfilter//dt).spline(smoothingfactor=splinefactor, strict=False))
        elif smoothing is None:
            return vc
        else:
            raise ValueError('Unknown smoothing "%s"' % str(smoothing))


class ActorProposalRefinement(VideoProposalRefinement):
    """Only refine the primary actor and nothing else"""
    pass

class ActorAssociation(VideoProposalRefinement):
    """pycollector.detection.VideoAssociation() class
       
       Select the best object proposal track of the target class associated with the primary actor class by gated spatial IOU and cover.
       Add the best object track to the scene and associate with all activities performed by the primary actor.
    """
    def __call__(self, v, target, dt=3):
        assert target.lower() in self.allowable_objects(), "Actor Association must be to an allowable target class '%s'" % str(self.allowable_objects())
        assert len(v.objectlabels()) == 1, "Actor Association can only be performed with scenes containing a single actor"
        
        #va = super().__call__(v.clone(), dt=dt)  # tight proposal for primary actor (same keys)
        va = v.clone()  # assume tight proposal for primary actor already (Proposals already generated)
        vi = va.clone(rekey=True).meanmask().savetmp() if target in v.objectlabels() else None
        vp = super().__call__(vi if vi is not None else va.clone(rekey=True), dt=dt, miniou=0, mincover=0, byclass=True, refinedclass=target, dilate_height=4.0, dilate_width=16.0, pdist=True)  # close proposal for associated object (primary object blurred)
        vc = va.clone()  # for idempotence (same keys as v)
        for t in vp.tracklist():
            vc.activitymap(lambda a: a.add(t) if a.during_interval(t.startframe(), t.endframe()) else a).add(t)
        if vi is not None:
            os.remove(vi.filename())  # cleanup temporary masked video
        return vc

    
def _collectorproposal_vs_objectproposal(v, dt=1, miniou=0.2, smoothing='spline'):
    """Return demo video that compares the human collector annotated proposal vs. the ML annotated proposal for a vipy.video.Scene()"""
    assert isinstance(v, vipy.video.Scene)
    v_human = v.clone().trackmap(lambda t: t.shortlabel('%s (collector box)' % t.category()))
    v_object = v.clone().trackmap(lambda t: t.shortlabel('%s (ML box)' % t.category()))
    return VideoProposalRefinement()(v_object, dt=dt, miniou=miniou, smoothing=smoothing).union(v_human, spatial_iou_threshold=1)

