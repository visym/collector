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
import pycollector.model.yolov5.models.yolo


class TorchNet(object):

    def gpu(self, idlist, batchsize=None):
        assert batchsize is None or (isinstance(batchsize, int) and batchsize > 0), "Batchsize must be integer"
        assert idlist is None or isinstance(idlist, int) or (isinstance(idlist, list) and len(idlist)>0), "Input must be a non-empty list of integer GPU ids"
        self._batchsize = int(batchsize if batchsize is not None else (self._batchsize if hasattr(self, '_batchsize') else 1))

        idlist = tolist(idlist)
        self._devices = ['cuda:%d' % k if k is not None and torch.cuda.is_available() and k != 'cpu' else 'cpu' for k in idlist]
        #self._tensortype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor       
        self._tensortype = torch.FloatTensor       

        if not hasattr(self, '_gpulist') or not hasattr(self, '_models') or idlist != self._gpulist  or not hasattr(self, '_models'):        
            self._models = [copy.deepcopy(self._model).to(d, non_blocking=False) for d in self._devices]
            for (d,m) in zip(self._devices, self._models):
                m.eval()
            self._gpulist = idlist
        torch.set_grad_enabled(False)
        return self

    def cpu(self, batchsize=None):
        return self.gpu(idlist=['cpu'], batchsize=batchsize)
    
    def iscpu(self):
        return any(['cpu' in d for d in self._devices])

    def isgpu(self):
        return any(['cuda' in d for d in self._devices])
    
    def __call__(self, t):
        """Parallel evaluation of tensor to split across GPUs set up in gpu().  t should be of size (ngpu*batchsize)
        
           * Note: Do not use DataParallel, this replicates the multi-gpu batch on device 0 and results in out of memory
        """
        assert len(t) <= self.batchsize(), "Invalid batch size"
        todevice = [b.pin_memory().to(d, non_blocking=True) for (b,d) in zip(t.split(self._batchsize) , self._devices)]  # async?
        fromdevice = [m(b) for (m,b) in zip(self._models, todevice)]   # async?
        return torch.cat([r.detach().cpu() for r in fromdevice], dim=0)
        
    def batchsize(self):
        return int(len(self._models)*self._batchsize)
        

class FaceDetector(TorchNet):
    """Faster R-CNN based face detector
    
    """

    def __init__(self, weightfile=None, gpu=None):    
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'face')

        weightfile = os.path.join(indir, 'resnet-101_faster_rcnn_ohem_iter_20000.pth') if weightfile is None else weightfile
        if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, 'a759030540a4a5284baa93d3ef5e47ed40cae6d6'):
            print('[pycollector.detection]: Downloading face detector weights ...')
            os.system('wget -c https://dl.dropboxusercontent.com/s/rdfre0oc456t5ee/resnet-101_faster_rcnn_ohem_iter_20000.pth -O %s' % weightfile)  # FIXME: replace with better solution
        assert vipy.downloader.verify_sha1(weightfile, 'a759030540a4a5284baa93d3ef5e47ed40cae6d6'), "Face detector download failed with SHA1='%s'" % (vipy.downloader.generate_sha1(weightfile))
        self._model = FaceRCNN(model_path=weightfile)
        #self._model.eval()  # Set in evaluation mode

        #if gpu is not None:
        #    self.gpu(gpu, batchsize)
        #else:
        #    self.cpu()
        
    def __call__(self, im):
        assert isinstance(im, vipy.image.Image)
        return vipy.image.Scene(array=im.numpy(), colorspace=im.colorspace(), objects=[vipy.object.Detection('face', xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3], confidence=bb[4]) for bb in self._model(im)]).union(im)

    def batchsize(self):
        return 1  # FIXME

    
class Yolov5(TorchNet):
    """Yolov5 based object detector

       >>> d = pycollector.detection.Detector()
       >>> d(vipy.image.vehicles()).show()

    """
    
    def __init__(self, batchsize=1, weightfile=None, gpu=None):    
        self._mindim = 640  # must be square
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'yolov5')
        cfgfile = os.path.join(indir, 'models', 'yolov5x.yaml')        
        weightfile = os.path.join(indir, 'yolov5x.weights') if weightfile is None else weightfile
        if not os.path.exists(weightfile):
            print('[pycollector.detection]: Downloading weights ...')
            os.system('wget -c https://dl.dropboxusercontent.com/s/jcwvz9ncjwpoat0/yolov5x.weights -O %s' % weightfile)  # FIXME: replace with better solution
            assert vipy.downloader.verify_sha1(weightfile, 'bdf2f9e0ac7b4d1cee5671f794f289e636c8d7d4'), "Object detector download failed"

        # First import: load yolov5x.pt, disable fuse() in attempt_load(), save state_dict weights and load into newly pathed model
        with torch.no_grad():
            self._model = pycollector.model.yolov5.models.yolo.Model(cfgfile, 3, 80)
            self._model.load_state_dict(torch.load(weightfile))
            self._model.fuse()
            self._model.eval()

        self._models = [self._model]
        
        self._batchsize = batchsize        
        assert isinstance(self._batchsize, int), "Batchsize must be integer"
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self._index2cls = {k:c for (c,k) in self._cls2index.items()}

        self._device = None
        #self._gpulist = gpu  # will be set in self.gpu()
        if gpu is not None:
            self.gpu(gpu, batchsize)
        else:
            self.cpu()
        torch.set_grad_enabled(False)
        
    def __call__(self, imlist, conf=1E-3, iou=0.5, union=False, objects=None):
        """Run detection on an image list at specific mininum confidence and iou NMS

           - yolov5 likes to split people into upper torso and lower body when in unfamilar poses (e.g. sitting, crouching)

        """
        assert isinstance(imlist, vipy.image.Image) or (isinstance(imlist, list) and all([isinstance(i, vipy.image.Image) for i in imlist])), "Invalid input - must be vipy.image.Image object and not '%s'" % (str(type(imlist)))
        assert objects is None or (isinstance(objects, list) and all([(k[0] if isinstance(k, tuple) else k) in self._cls2index for k in objects])), "Objects must be a list of allowable categories"
        objects = {(k[0] if isinstance(k,tuple) else k):(k[1] if isinstance(k,tuple) else k) for k in objects} if isinstance(objects, list) else objects

        with torch.no_grad():
            imlist = tolist(imlist)
            imlistdets = []
            t = torch.cat([im.clone(shallow=True).maxsquare().mindim(self._mindim).gain(1.0/255.0).torch(order='NCHW') for im in imlist])  # triggers load
            if torch.cuda.is_available() and not self.iscpu():
                t = t.pin_memory()

            assert len(t) <= self.batchsize(), "Invalid batch size: %d > %d" % (len(t), self.batchsize())
            todevice = [b.to(d, memory_format=torch.contiguous_format, non_blocking=True) for (b,d) in zip(t.split(self._batchsize), self._devices)]  # contiguous_format required for torch-1.8.1
            fromdevice = [m(b)[0] for (m,b) in zip(self._models, todevice)]     # detection
        
            t_out = [torch.squeeze(t, dim=0) for d in fromdevice for t in torch.split(d, 1, 0)]   # unpack batch to list of detections per imag
            t_out = [torch.cat((t[:,0:5], torch.argmax(t[:,5:], dim=1, keepdim=True)), dim=1) for t in t_out]  # filter argmax on device 
            t_out = [t[t[:,4]>conf].cpu().detach().numpy() for t in t_out]  # filter conf on device (this must be last)

        k_valid_objects = set([self._cls2index[k] for k in objects.keys()]) if objects is not None else self._cls2index.values()        
        for (im, dets) in zip(imlist, t_out):
            if len(dets) > 0:
                k_det = np.argwhere((dets[:,4] > conf).flatten() & np.array([int(d) in k_valid_objects for d in dets[:,5]])).flatten().tolist()
                objectlist = [vipy.object.Detection(xcentroid=float(dets[k][0]),
                                                    ycentroid=float(dets[k][1]),
                                                    width=float(dets[k][2]),
                                                    height=float(dets[k][3]),
                                                    confidence=float(dets[k][4]),
                                                    category='%s' % self._index2cls[int(dets[k][5])],
                                                    id=True)
                              for k in k_det]
                                 
                scale = max(im.shape()) / float(self._mindim)  # to undo
                objectlist = [obj.rescale(scale) for obj in objectlist]
                objectlist = [obj.category(objects[obj.category()]) if objects is not None else obj for obj in objectlist]  # convert to target class before NMS
            else:
                objectlist = []

            imd = im.objects(objectlist) if not union else im.objects(objectlist + im.objects())
            if iou > 0:
                imd = imd.nms(conf, iou)  
            imlistdets.append(imd)  
            
        return imlistdets if self._batchsize > 1 else imlistdets[0]

    def classlist(self):
        return list(self._cls2index.keys())
    
    
class Yolov3(TorchNet):
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
        assert objects is None or (isinstance(objects, list) and all([(k[0] if isinstance(k, tuple) else k) in self._cls2index for k in objects])), "Objects must be a list of allowable categories"
        objects = {(k[0] if isinstance(k,tuple) else k):(k[1] if isinstance(k,tuple) else k) for k in objects} if isinstance(objects, list) else objects

        imlist = tolist(im)
        imlistdets = []
        t = torch.cat([im.clone().maxsquare().mindim(self._mindim).gain(1.0/255.0).torch(order='NCHW') for im in imlist]).type(self._tensortype)  # triggers load
        t_out = super().__call__(t).detach().numpy()   # parallel multi-GPU evaluation, using TorchNet()
        for (im, dets) in zip(imlist, t_out):
            k_class = np.argmax(dets[:,5:], axis=1).flatten().tolist()
            k_det = np.argwhere((dets[:,4] > conf).flatten() & np.array([((objects is None) or (self._index2cls[k] in objects.keys())) for k in k_class])).flatten().tolist()
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
            objectlist = [obj.category(objects[obj.category()]) if objects is not None else obj for obj in objectlist]
            imd = im.clone().array(im.numpy()).objects(objectlist).nms(conf, iou)  # clone for shared attributese
            imlistdets.append(imd if not union else imd.union(im))
            
        return imlistdets if self._batchsize > 1 else imlistdets[0]

    def classlist(self):
        return list(self._cls2index.keys())
    

class ObjectDetector(Yolov5):
    """Default object detector"""
    pass


class MultiscaleObjectDetector(ObjectDetector):  
    """Given a list of images, break each one into a set of overlapping tiles, and ObjectDetector() on each, then recombining detections"""
    def __call__(self, imlist, conf=0.5, iou=0.5, maxarea=1.0, objects=None, overlapfrac=6, filterborder=True, cover=0.7):  
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(imlist, vipy.image.Image) or isinstance(imlist, list) and all([isinstance(im, vipy.image.Image) for im in imlist]), "invalid input"
        imlist = tolist(imlist)
        scale = imlist[0].mindim() / n
        
        (imlist_multiscale, imlist_multiscale_flat, n_coarse, n_fine) = ([], [], [], [])
        for im in imlist:
            imcoarse = [im]

            # FIXME: generalize this parameterization
            if overlapfrac == 6:
                imfine = (im.tile(n, n, overlaprows=im.height()-n, overlapcols=(3*n-im.width())//2) if (im.mindim()>=n and im.mindim() == im.height()) else
                          (im.tile(n, n, overlapcols=im.width()-n, overlaprows=(3*n-im.height())//2) if im.mindim()>=n else []))  # 2x3 tile, assumes im.mindim() == (n+n/2)
                if len(imfine) != 6:
                    print('WARNING: len(imtile) = %d for overlapfrac = %d' % (len(imfine), overlapfrac))  # Sanity check                    
                    
            elif overlapfrac == 2:
                imfine = (im.tile(n, n, overlaprows=0, overlapcols=(2*n-im.width())//2) if (im.mindim()>=n and im.mindim() == im.height()) else
                          (im.tile(n, n, overlapcols=0, overlaprows=(2*n-im.height())//2) if im.mindim()>=n else []))  # 1x2 tile, assumes im.mindim() == (n)
                if len(imfine) != 2:
                    print('WARNING: len(imtile) = %d for overlapfrac = %d' % (len(imfine), overlapfrac))  # Sanity check
                    
            elif overlapfrac == 0:
                imfine = []
                
            else:
                raise
            # /FIXME
            
            n_coarse.append(len(imcoarse))
            n_fine.append(len(imfine))
            imlist_multiscale.append(imcoarse+imfine)
            imlist_multiscale_flat.extend(imcoarse + [imf.maxsquare(n) for imf in imfine])            

        imlistdet_multiscale_flat = [im for iml in chunklistbysize(imlist_multiscale_flat, self.batchsize()) for im in tolist(f(iml, conf=conf, iou=0, objects=objects))]
        
        imlistdet = []
        for (k, (iml, imb, nf, nc)) in enumerate(zip(imlist, imlist_multiscale, n_fine, n_coarse)):
            im_multiscale = imlistdet_multiscale_flat[0:nf+nc]; imlistdet_multiscale_flat = imlistdet_multiscale_flat[nf+nc:];
            imcoarsedet = im_multiscale[0].mindim(iml.mindim())
            imcoarsedet_imagebox = imcoarsedet.imagebox()
            if filterborder:
                imfinedet = [im.nms(conf, iou, cover=cover).objectfilter(lambda o: ((maxarea==1 or (o.area()<=maxarea*im.area())) and   # not too big relative to tile
                                                                                    ((o.isinterior(im.width(), im.height(), border=0.9) or  # not occluded by any tile boundary 
                                                                                      o.clone().dilatepx(0.1*im.width()+1).cover(im.imagebox()) == o.clone().dilatepx(0.1*im.width()+1).set_origin(im.attributes['tile']['crop']).cover(imcoarsedet_imagebox)))))  # or only occluded by image boundary
                             for im in im_multiscale[nc:]]
                imfinedet = [im.objectmap(lambda o: o.set_origin(im.attributes['tile']['crop'])) for im in imfinedet]  # shift objects only, equivalent to untile() but faster
                imcoarsedet = imcoarsedet.objects( imcoarsedet.objects() + [o for im in imfinedet for o in im.objects()])  # union
            else:
                imfinedet = iml.untile( im_multiscale[nc:] )
                imcoarsedet = imcoarsedet.union(imfinedet) if imfinedet is not None else imcoarsedet
            imlistdet.append(imcoarsedet.nms(conf, iou, cover=cover))

        return imlistdet[0] if len(imlistdet) == 1 else imlistdet

    
class VideoDetector(ObjectDetector):  
    """Iterate ObjectDetector() over each frame of video, yielding the detected frame"""
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"        
        for im in v.stream():
            yield super().__call__(im, conf=conf, iou=iou)

                        
class MultiscaleVideoDetector(MultiscaleObjectDetector):
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"
        for imf in v.stream():
            yield super().__call__(imf, conf, iou)


class VideoTracker(ObjectDetector):
    def __call__(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05):
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(v, vipy.video.Video), "Invalid input"
        assert objects is None or all([o in self.classlist() for o in objects]), "Invalid object list"
        vc = v.clone().clear()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate(tolist(f(vb.framelist(), minconf, miniou, union=False, objects=objects))):
                yield vc.assign(k*self.batchsize()+j, im.clone().objects(), minconf=trackconf, maxhistory=maxhistory)  # in-place            

    def track(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05, verbose=False):
        """Batch tracking"""
        for (k,vt) in enumerate(self.__call__(v.clone(), minconf=minconf, miniou=miniou, maxhistory=maxhistory, smoothing=smoothing, objects=objects, trackconf=trackconf)):
            if verbose:
                print('[pycollector.detection.VideoTracker][%d]: %s' % (k, str(vt)))  
        return vt


class FaceTracker(FaceDetector):
    def __call__(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, trackconf=0.05):
        (f) = (super().__call__)
        assert isinstance(v, vipy.video.Video), "Invalid input"
        vc = v.clone().clear()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate([f(im) for im in vb.framelist()]):
                yield vc.assign(k*self.batchsize()+j, im.clone().objects(), minconf=trackconf, maxhistory=maxhistory)  # in-place            

    def track(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05, verbose=False):
        """Batch tracking"""
        for (k,vt) in enumerate(self.__call__(v.clone(), minconf=minconf, miniou=miniou, maxhistory=maxhistory, smoothing=smoothing, trackconf=trackconf)):
            if verbose:
                print('[pycollector.detection.FaceTracker][%d]: %s' % (k, str(vt)))  
        return vt
    
    
class MultiscaleVideoTracker(MultiscaleObjectDetector):
    """MultiscaleVideoTracker() class"""

    def __init__(self, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05, verbose=False, gpu=None, batchsize=1, weightfile=None, overlapfrac=2, detbatchsize=None, gate=0):
        super().__init__(gpu=gpu, batchsize=batchsize, weightfile=weightfile)
        self._minconf = minconf
        self._miniou = miniou
        self._maxhistory = maxhistory
        self._smoothing = smoothing
        self._objects = objects
        self._trackconf = trackconf
        self._verbose = verbose
        self._maxarea = 1.0
        self._overlapfrac = overlapfrac
        self._detbatchsize = detbatchsize if detbatchsize is not None else self.batchsize()
        self._gate = gate

    def _track(self, vi, stride=1):
        """Yield vipy.video.Scene(), an incremental tracked result for each frame.
        """
        assert isinstance(vi, vipy.video.Video), "Invalid input"

        (det, n) = (super().__call__, self._mindim)
        for (k, vb) in enumerate(vi.stream().batch(self._detbatchsize)):
            framelist = vb.framelist()
            for (j, im) in zip(range(0, len(framelist), stride), tolist(det(framelist[::stride], self._minconf, self._miniou, self._maxarea, objects=self._objects, overlapfrac=self._overlapfrac))):
                for i in range(j, j+stride):                    
                    if i < len(framelist):
                        yield (vi.assign(k*self._detbatchsize+i, im.objects(), minconf=self._trackconf, maxhistory=self._maxhistory, gate=self._gate) if (i == j) else vi)

    def __call__(self, vi, stride=1):
        return self._track(vi, stride)
    
    def stream(self, vi):
        return self._track(vi)

    def track(self, vi, verbose=False):
        """Batch tracking"""
        for v in self.stream(vi):
            if verbose:
                print(vi)
        return vi
        


class Proposal(ObjectDetector):
    def __call__(self, v, conf=1E-2, iou=0.8):
        return super().__call__(v, conf, iou)
    
    
class VideoProposal(Proposal):
    """pycollector.detection.VideoProposal() class.
    
       Track-based object proposals in video.
    """
    def allowable_objects(self):
        return ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle', 'motorbike', 'truck']

    def isallowable(self, v):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be vipy.video.Video not '%s'" % (str(type(v)))
        return all([c.lower() in self.allowable_objects() for c in v.objectlabels()]) # for now

    def __call__(self, v, conf=1E-2, iou=0.8, dt=1, target=None, activitybox=False, dilate=4.0, dilate_height=None, dilate_width=None):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be vipy.video.Video not '%s'" % (str(type(v)))

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

        for i in range(0, len(tensor), self.batchsize()):
            with torch.no_grad():
                t = tensor[i:i+self.batchsize()]
                todevice = [b.to(d, non_blocking=True) for (b,d) in zip(t.split(self._batchsize) , self._devices)]  # async?
                fromdevice = [m(b)[0] for (m,b) in zip(self._models, todevice)]     # detection!
                dets = [torch.squeeze(t, dim=0).cpu().detach().numpy() for d in fromdevice for t in torch.split(d, 1, 0)]   # unpack batch to list of detections per imag

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


class FaceProposalRefinement():
    def __call__(self, vp, vt, spatial_iou_threshold=0.2):
        assert isinstance(vp, vipy.video.Scene) and isinstance(vt, vipy.video.Scene)        
        assert len(vp.tracklist()) == 1 and all([t.category().lower() in ['face'] for t in vp.tracklist()])
        assert len(vt.tracklist()) >= 0 and all([t.category().lower() in ['face'] for t in vt.tracklist()])

        vc = vt.clone()
        suppressed = set([])
        for (ti,s,c) in sorted([(t, vp.actor().segment_percentileiou(t, percentile=0.5), t.confidence()) for t in vc.tracklist()], key=lambda x: x[1]*x[2], reverse=True):
            if ti.id() not in suppressed and s > spatial_iou_threshold and ti.category() == vp.actor().category():
                # Assign proposal to best track
                for a in vp.activitylist():
                    if a.hastrack(vp.actor()) and not vc.hasactivity(a.id()):
                        vc.activities()[a.id()] = a.clone().replace(vp.actor(), ti)
                    elif a.hastrack(vp.actor()) and vc.hasactivity(a.id()):
                        vc.activities()[a.id()].append(ti)
                
                # Supress all other temporally overlapping tracks (not including ti)
                suppressed = suppressed.union([t.id() for t in vc.tracklist() if t.id() != ti.id() and t.temporal_distance(ti) == 0 and t.id() not in suppressed])
            else:
                suppressed.add(ti.id())        
                
        return vc.trackfilter(lambda t: t.id() not in suppressed)
        
        
class VideoProposalRefinement(VideoProposal):
    """pycollector.detection.VideoProposalRefinement() class.
    
       Track-based object proposal refinement of a weakly supervised loose object box from a human annotator.
    """
    
    def __call__(self, v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=1, meanfilter=15, mincover=0.8, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True, byclass=True, dilate_height=None, dilate_width=None, refinedclass=None, pdist=False, minconf=1E-2):
        """Replace proposal in v by best (maximum overlap and confidence) object proposal in vc.  If no proposal exists, delete the proposal."""
        assert isinstance(v, vipy.video.Scene), "Invalid input - must be vipy.video.Scene not '%s'" % (str(type(v)))
        assert not (byclass is False and refinedclass is not None), "Invalid input"
        
        if not self.isallowable(v):
            warnings.warn("Invalid object labels '%s' for proposal, must be only one target object label and must be in '%s' - returning original video" % (str(v.objectlabels()), str(self.allowable_objects())))
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


class ActorAssociation(MultiscaleVideoTracker):
    """pycollector.detection.VideoAssociation() class
       
       Select the best object track of the target class associated with the primary actor class by gated spatial IOU and distance.
       Add the best object track to the scene and associate with all activities performed by the primary actor.
    """

    @staticmethod
    def isallowable(v, actor_class, association_class, fps=None):
        allowable_objects = ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle', 'motorbike', 'truck']        
        return (actor_class.lower() in allowable_objects and
                all([a.lower() in allowable_objects for a in vipy.util.tolist(association_class)]) and
                actor_class.lower() in v.objectlabels(lower=True))
        

    def __call__(self, v, actor_class, association_class, fps=None, dilate=2.0, activity_class=None, maxcover=0.8, max_associations=1, min_confidence=0.4):
        allowable_objects = ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle', 'motorbike', 'truck']        
        association_class = [a.lower() for a in vipy.util.tolist(association_class)]
        assert actor_class.lower() in allowable_objects, "Primary Actor '%s' not in allowable target class '%s'" % (actor_class.lower(), str(allowable_objects))
        assert all([a in allowable_objects for a in allowable_objects]), "Actor Association '%s' not in allowable target class '%s'" % (str(association_class), str(allowable_objects))
        assert actor_class.lower() in v.objectlabels(lower=True), "Actor Association can only be performed with scenes containing an allowable actor not '%s'" % str(v.objectlabels())
        
        # Track objects
        vc = v.clone()
        if fps is not None:
            for t in vc.tracks().values():
                t._framerate = v.framerate()  # HACK: backwards compatibility
            for a in vc.activities().values():
                a._framerate = v.framerate()  # HACK: backwards compatibility
        vc = vc.framerate(fps) if fps is not None else vc   # downsample
        vt = self.track(vc.clone())  # track at downsampled framerate

        # Actor assignment: for every activity, find track with best target object assignment to actor (first track in video)
        for a in vc.activities().values():
            candidates = [t for t in vt.tracks().values() if (t.category().lower() in association_class and
                                                              t.during_interval(a.startframe(), a.endframe()) and
                                                              t.confidence() > min_confidence and  # must have minimum confidence
                                                              (actor_class.lower() not in association_class or t.segmentcover(vc.actor()) < maxcover) and
                                                              vc.actor().boundingbox().dilate(dilate).hasintersection(t.boundingbox()))] # candidate assignment (cannot be actor, or too far from actor)
            if len(candidates) > 0:
                # best assignment is track closest to actor with maximum confidence and minimum dilated overlap
                trackconf = sorted([(t, vc.actor().boundingbox().dilate(dilate).iou(t.boundingbox()) * t.confidence()) for t in candidates], key=lambda x: x[1], reverse=True)
                for (t, conf) in trackconf[0:max_associations]:
                    if a.during_interval(t.startframe(), t.endframe()) and activity_class is None or a.category() == activity_class:
                        a.add(t)
                        vc.add(t)

        return vc.framerate(v.framerate()) if vc.framerate() != v.framerate() else vc   # upsample

    
def _collectorproposal_vs_objectproposal(v, dt=1, miniou=0.2, smoothing='spline'):
    """Return demo video that compares the human collector annotated proposal vs. the ML annotated proposal for a vipy.video.Scene()"""
    assert isinstance(v, vipy.video.Scene)
    v_human = v.clone().trackmap(lambda t: t.shortlabel('%s (collector box)' % t.category()))
    v_object = v.clone().trackmap(lambda t: t.shortlabel('%s (ML box)' % t.category()))
    return VideoProposalRefinement()(v_object, dt=dt, miniou=miniou, smoothing=smoothing).union(v_human, spatial_iou_threshold=1)

