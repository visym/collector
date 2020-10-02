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
from pycollector.model.face.detection import FaceRCNN 


class TorchNet(object):
    def gpu(self, k):
        deviceid = 'cuda:%d' % k if torch.cuda.is_available() and k is not None else 'cpu'
        device = torch.device(deviceid)
        self._tensortype = torch.cuda.FloatTensor if deviceid != 'cpu' and torch.cuda.is_available() else torch.FloatTensor        
        self._model = self._model.to(device)
        self._model.eval()  # Set in evaluation mode
        self._device = device
        return self


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
        return vipy.image.Scene(array=im.numpy(), colorspace=im.colorspace(), objects=[vipy.object.Detection('face', xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3], confidence=bb[4]) for bb in self._model(im)])


class Detector(object):
    """Yolov3 based object detector

       >>> d = pycollector.detection.Detector()
       >>> d(vipy.image.vehicles()).show()

    """
    
    def __init__(self, batchsize=1, weightfile=None):    
        self._mindim = 416
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
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self._index2cls = {k:c for (c,k) in self._cls2index.items()}
        self.gpu(vipy.globals.gpuindex())

    def __call__(self, im, conf=5E-1, iou=0.5):
        assert isinstance(im, vipy.image.Image), "Invalid input - must be vipy.image.Image object and not '%s'" % (str(type(im)))
        self.gpu(vipy.globals.gpuindex())

        scale = max(im.shape()) / float(self._mindim)  # to undo
        t = im.clone().maxsquare().mindim(self._mindim).mat2gray().torch().type(self._tensortype).to(self._device)
        dets = self._model(t)[0]
        objects = [vipy.object.Detection(xcentroid=float(d[0]),
                                         ycentroid=float(d[1]),
                                         width=float(d[2]),
                                         height=float(d[3]),
                                         confidence=float(d[4]),
                                         category='%s' % self._index2cls[int(np.argmax(d[5:]))])
                   for d in dets if float(d[4]) > conf]
        objects = [obj.rescale(scale) for obj in objects]
        return vipy.image.Scene(array=im.numpy(), objects=objects).nms(conf, iou)

    def classlist(self):
        return list(self._cls2index.keys())
    
    def gpu(self, k):
        deviceid = 'cuda:%d' % k if torch.cuda.is_available() and k is not None else 'cpu'
        device = torch.device(deviceid)
        self._tensortype = torch.cuda.FloatTensor if deviceid != 'cpu' and torch.cuda.is_available() else torch.FloatTensor        
        self._model = self._model.to(device)
        self._model.eval()  # Set in evaluation mode
        self._device = device
        return self


class VideoDetector(Detector):
    def __call__(v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video)
        raise NotImplementedError('Coming Soon')

    
class Proposal(Detector):
    def __call__(self, v, conf=1E-2, iou=0.8):
        return super(Proposal, self).__call__(v, conf, iou)

    
class VideoProposal(Proposal):
    def __call__(self, v, conf=1E-2, iou=0.8, dt=1, target=None, activitybox=False, dilate=4.0):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be vipy.video.Video not '%s'" % (str(type(v)))
        self.gpu(vipy.globals.gpuindex())

        # Optional target class:  "Person" or "Vehicle" or "Car" or "Motorcycle" for now
        c = {'person':[self._cls2index['person']], 
             'vehicle':[self._cls2index['car'], self._cls2index['motorbike'], self._cls2index['truck']], 
             'car':[self._cls2index['car'], self._cls2index['truck']],
             'motorcycle':[self._cls2index['motorbike']]}
        assert target is None or (isinstance(target, list) and all([t in c.keys() for t in target]))

        # Parameters to undo
        bb = v.activitybox(dilate=dilate).imclipshape(v.width(), v.height()) if activitybox else vipy.geometry.imagebox(v.shape())
        scale = max(bb.shape()) / float(self._mindim)

        # Batched proposals on transformed video
        ims = []
        tensor = v.clone().crop(bb, zeropad=False).maxsquare().mindim(self._mindim).torch()  # NxCxHxW
        img = [v[k].numpy() for k in range(0, len(tensor), dt)]
        tensor = tensor[::dt]  # skip

        for i in range(0, len(tensor), self._batchsize):
            dets = self._model(tensor[i:i+self._batchsize].type(self._tensortype).to(self._device))  # copy here
            for (j,det) in enumerate(dets):
                # Objects in transformed video
                objs = [vipy.object.Detection(xcentroid=float(d[0]), 
                                              ycentroid=float(d[1]), 
                                              width=float(d[2]), 
                                              height=float(d[3]), 
                                              confidence=(float(d[4]) + (max([d[5+k] for t in target for k in c[t]]) if target is not None else 0)), 
                                              category='%1.1f' % float(d[4])) 
                        for d in det if (float(d[4]) > conf) and (target is None or (max([d[5+k] for t in target for k in c[t]]) > conf))]

                # Objects in source video
                objs = [obj.rescale(scale).translate(bb.xmin(), bb.ymin()) for obj in objs]
                ims.append( vipy.image.Scene(array=img[i+j], objects=objs).nms(conf, iou) )
        ims.append(ims[-1])  # add one more for inclusive track endpoints
        return ims

    
class VideoProposalRefinement(VideoProposal):
    def __call__(self, v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=1, meanfilter=15, mincover=0.8, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True, byclass=True):
        """Replace proposal in v by best (maximum overlap and confidence) object proposal in vc.  If no proposal exists, delete the proposal."""
        assert all([c.lower() in ['person', 'vehicle', 'car', 'motorcycle'] for c in v.objectlabels()])  # for now

        vp = super(VideoProposalRefinement, self).__call__(v, proposalconf, proposaliou, dt=dt, activitybox=True, dilate=4.0, target=[c.lower() for c in v.objectlabels()] if byclass else None)  # subsampled proposals
        vc = v.clone(rekey=True, flushforward=True, flushbackward=True).trackfilter(lambda t: len(t) > dt)
        for (ti, t) in vc.tracks().items():
            t.resample(dt=dt)  # interpolated keyframes for source proposal
            bbprev = None  # last box assignment
            s = shapeiou  # shape assignment threshold
            for (f, bb) in zip(t.clone().keyframes(), t.clone().keyboxes()):  # clone because vc is being modified in-loop
                fs = (f // dt)  # subsampled frame index
                if fs>=0 and fs<len(vp):  
                    # Assignment: maximum (overlap with previous box + same shape as previous box + overlap with human box) * (objectness confidence + class confidence)
                    # Assignment constraints: (new box must not be too small relative to collector box) and (new box must be mostly contained within the collector box) and (new box must mostly overlap previous box) 
                    assignment = sorted([(bbp, (((bbp.shapeiou(bbprev) + bbp.iou(bbprev)) if bbprev is not None else 1.0) + bb.iou(bbp))*bbp.confidence())
                                         for bbp in vp[fs].objects() 
                                         if (bb.iou(bbp)>miniou and bbp.cover(bb)>mincover and (bbprev is None or bbprev.shapeiou(bbp)>s))], key=lambda x: x[1])
                    if len(assignment) > 0:
                        (bbp, iou) = assignment[-1]  # best assignment
                        vc.tracks()[ti].replace(f, bbp.clone().category( t.category()) )        
                        bbprev = bbp.clone() # update last assignment
                        s = shapeiou
                    else:
                        if strict:
                            vc.tracks()[ti].delete(f)  # Delete proposal that has no object proposal, otherwise use source proposal for interpolation
                        s = max(0, s-(0.001*dt)) if (bbprev is not None and bb.iou(bbprev)>miniou and bbprev.cover(bb)>mincover) else 0  # gate increase for shape deformation, or reset if we lost it

        vc = vc.trackfilter(lambda t: len(t)>dt)  # remove empty tracks

        # Proposal smoothing
        if smoothing == 'mean':
            # Mean track smoothing: mean shape smoothing with mean coordinate smoothing with very small support for unstabilized video
            return vc.trackmap(lambda t: t.smoothshape(width=meanfilter//dt).smooth(3))
        elif smoothing == 'spline':
            # Cubic spline track smoothing with mean shape smoothing 
            return vc.trackmap(lambda t: t.smoothshape(width=meanfilter//dt).spline(smoothingfactor=splinefactor))
        elif smoothing is None:
            return vc
        else:
            raise ValueError('Unknown smoothing "%s"' % str(smoothing))


def collectorproposal_vs_objectproposal(v, dt=1, miniou=0.2, smoothing='spline'):
    """Return demo video that compares the human collector annotated proposal vs. the ML annotated proposal for a vipy.video.Scene()"""
    assert isinstance(v, vipy.video.Scene)
    v_human = v.clone().trackmap(lambda t: t.shortlabel('%s (collector box)' % t.category()))
    v_object = v.clone().trackmap(lambda t: t.shortlabel('%s (ML box)' % t.category()))
    return VideoProposalRefinement(batchsize=8)(v_object, dt=dt, miniou=miniou, smoothing=smoothing).union(v_human, spatial_iou_threshold=1)

