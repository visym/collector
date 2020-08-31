import os
import sys
import torch
import vipy
import shutil
from vipy.globals import print
from vipy.util import remkdir, filetail, readlist, tolist, filepath
from pycollector.video import Video
from pycollector.yolov3.models import Darknet


class Proposal(object):
    def __init__(self, batchsize=1, weightfile=None):
        self._mindim = 416
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'yolov3')
        weightfile = os.path.join(indir, 'yolov3.weights') if weightfile is None else weightfile
        cfgfile = os.path.join(indir, 'yolov3.cfg')
        self._model = Darknet(cfgfile, img_size=self._mindim)
        if not os.path.exists(weightfile) or not vipy.downloader.verify_sha1(weightfile, '520878f12e97cf820529daea502acca380f1cb8e'):
            #vipy.downloader.download('https://www.dropbox.com/s/ve9cpuozbxh601r/yolov3.weights', os.path.join(indir, 'yolov3.weights'))
            print('[pycollector.detection]: Downloading object detector weights ...')
            os.system('wget -c https://www.dropbox.com/s/ve9cpuozbxh601r/yolov3.weights -O %s' % weightfile)  # FIXME: replace with better solution
        assert vipy.downloader.verify_sha1(weightfile, '520878f12e97cf820529daea502acca380f1cb8e'), "Object detector download failed"
        self._model.load_darknet_weights(weightfile)
        self._model.eval()  # Set in evaluation mode
        self._batchsize = batchsize        
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self.gpu(vipy.globals.gpuindex())

    def __call__(self, im, conf=1E-1, iou=0.8):
        assert isinstance(im, vipy.image.Image)
        self.gpu(vipy.globals.gpuindex())

        scale = max(im.shape()) / float(self._mindim)  # to undo
        t = im.clone().maxsquare().mindim(self._mindim).mat2gray().torch().type(self._tensortype).to(self._device)
        dets = self._model(t)[0]
        objects = [vipy.object.Detection(xcentroid=float(d[0]), ycentroid=float(d[1]), width=float(d[2]), height=float(d[3]), confidence=float(d[4]), category='%1.1f' % float(d[4])) for d in dets if float(d[4]) > conf]
        objects = [obj.rescale(scale) for obj in objects]
        return vipy.image.Scene(array=im.numpy(), objects=objects).nms(conf, iou)
        
    def gpu(self, k):
        deviceid = 'cuda:%d' % k if torch.cuda.is_available() and k is not None else 'cpu'
        device = torch.device(deviceid)
        self._tensortype = torch.cuda.FloatTensor if deviceid != 'cpu' and torch.cuda.is_available() else torch.FloatTensor        
        self._model = self._model.to(device)
        self._model.eval()  # Set in evaluation mode
        self._device = device
        return self


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


def _test_proposal():
    videoids = ['20200522_0150473561208997437622670',
                '00EBA2AD-8D45-4888-AA28-A71FAAA1ED88-1000-0000005BA4C1929E',
                'F3EAB9E5-3DCF-41AD-90C0-626E56E367A9-1000-00000064E17CA24F',
                'DD4F0E6E-A222-4FE0-8180-378B57A9FA76-2093-0000019498261CC0',
                '20200528_1411011390426986',
                '20200528_141345-2035479356',
                '20200520_2017063615715963202368863',
                '477AE69F-153D-48F7-8CEA-CE54688DE942-8899-0000068727516F05',
                'DCE8A056-8EAF-4268-8624-5FA4EB42B416-4214-00000259540C3B4C',
                '20200521_180041290232562',
                '20200526_183503604561539675548030',
                '9F7BEDDF-4317-4CCF-A17B-D0CD84BE7D29-14557-000009A78377B03B',
                '82DD0A37-30CC-4A74-B1F9-028FDF567983-289-0000000E8DC9A679',
                'E4B58A6B-F79A-4E11-83C3-924BEDA69D3A-320-000000225CC6B4E8',
                '20200503_1101496154439725041568313',
                '20200505_1608442273346368432213366',
                '9802EE07-1C5F-467E-AF19-3569A6AF9440-1763-00000147FBD3138A',
                '20200423_1104206253700009606748525',
                '07826D93-E5C4-41DB-A8BC-4D3203E64F91-862-0000009D976888CB',
                '24FD34F3-AC56-4528-8770-D6A0A30A3358-4367-000002F2F18C0C5F',
                '6A8698F4-31C4-43E2-B061-55FF4E250615-264-0000000DE8AE00DB',
                '20200525_1830282647560902514919243',
                '20200525_1748021531575967472321212',
                '20200525_1658548039717461873489470',
                '133BA88D-A828-4397-81BD-6EEB9393F20B-710-0000005AEDD91457']
    
    
    shutil.rmtree(remkdir('test_proposal'))
    vipy.globals.gpuindex(0)
    for videoid in videoids:
        collectorproposal_vs_objectproposal(Video(videoid), dt=3).annotate().saveas(os.path.join(remkdir('test_proposal'), '%s.mp4' % videoid))
