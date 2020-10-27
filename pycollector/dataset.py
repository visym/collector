import os
import numpy as np
import pycollector.detection
from pycollector.globals import print
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, listpkl, listext
import random
import vipy
import vipy.util
import shutil


def tocsv(pklfile):
    """Convert a dataset to a standalne CSV file"""
    pass


def disjoint_activities(V, activitylist):    
    assert all([isinstance(v, vipy.video.Video) for v in V])
    assert isinstance(activitylist, list) and len(activitylist)>0 and len(activitylist[0]) == 2
    for (after, before) in activitylist:
        V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == after]) if a.category() == before else a) for v in V]  
    V = [v.activityfilter(lambda a: len(a)>0) for v in V]  # some activities may be zero length after disjoint
    return V


def reference(V, outdir, mindim=512):
    assert isinstance(V, list) and all([isinstance(v, vipy.video.Video) for v in V]), "Invalid input"
    
    V_dist = [(v, mindim, vipy.util.repath(v.filename(), filepath(v.filename()), vipy.util.remkdir(outdir))) for v in V]
    return (Batch(V_dist, strict=False)
            .map(lambda x: x[0].mindim(x[1]).saveas(x[2], flush=True).print())
            .result()) 

def activityclip(V, outdir, subdir='videos'):
    """Dataset filename structure: /outdir/$SUBDIR/$ACTIVITY_CATEGORY/$VIDEOID_$ACTIVITYINDEX.mp4"""
    activitylist = [a for v in V for a in v.activityclip()]
    V_dist = [(a, os.path.join(remkdir(outdir), subdir, a.category(), '%s_%d%s' % (filebase(a.filename()), k, mediaextension(a.filename())))) for (k,a) in activitylist]
    return (Batch(V_dist, strict=False)
            .map(lambda x: x[0].filename(x[1], copy=True).print())
            .result())

def objectclip(V, outdir, subdir='videos'):
    """Dataset filename structure: /outdir/$SUBDIR/$OBJECT_CATEGORY/$VIDEOID.mp4"""     
    tracklist = [t for v in V for t in v.tracksplit()]
    V_dist = [(t, os.path.join(remkdir(outdir), subdir, t.category(), filetail(t))) for t in tracklist]
    return (Batch(V_dist, strict=False)
            .map(lambda x: x[0].filename(x[1], copy=True).print())
            .result())

def subjectclip(V, outdir, subdir='videos'):
    """Dataset filename structure: /outdir/$SUBDIR/$SUBJECT_ID/$VIDEOID.mp4"""
    tracklist = [t for v in V for t in v.trackclip()]
    V_dist = [(t, os.path.join(remkdir(outdir), subdir, t.subjectid(), filetail(t))) for t in tracklist]
    return (Batch(V_dist, strict=False)
            .map(lambda x: x[0].filename(x[1], copy=True).print())
            .result())

def stabilize(V, outdir=None):
    assert vipy.version.is_at_least('1.8.33')
    assert all([isinstance(v, vipy.video.Video) for v in V]), "Invalid input"
    from vipy.flow import Flow
    from vipy.batch import Batch
    V_dist = [(v, vipy.util.repath(v.filename(), filepath(v.filename(), depth=2), vipy.util.remkdir(outdir)) if outdir is not None else v.filename()) for v in V]
    return (Batch(V_dist, strict=False)
            .filter(lambda x: x[0].canload())
            .map(lambda x: Flow(flowdim=256).stabilize(x[0], residual=True).saveas(x[1], flush=True).print())
            .filter(lambda x: (x is not None) and (not x.hasattribute('unstabilized'))) # remove unstabilized and failed
            .result())  


def resize_dataset(indir, outdir, dilate=1.2, maxdim=256, maxsquare=True):
    """Convert redistributable dataset by resizing the activity tube.
    
       * Indir: /path/to/src that contains trainset.pkl
       * outdir: should be /mypath/src

    """

    (indir, outdir) = (os.path.normpath(indir), os.path.normpath(outdir))
    print('[pycollector.dataset]: converting %s -> %s' % (indir, outdir))
    pklfiles = [f for f in findpkl(indir) if filebase(f) in set(['trainset','valset','testset'])]
    assert len(pklfiles) > 0
    for pklfile in pklfiles:
        b = vipy.batch.Batch(vipy.util.load(pklfile), 24)
        dataset = b.filter(lambda v: v.hastracks()).map(lambda v: v.activitytube(dilate=dilate, maxdim=maxdim).saveas(v.filename().replace(indir, outdir)).print()).result()
        vipy.util.distsave(dataset, datapath=indir, outfile=pklfile.replace(indir, outdir))
    return outdir


def boundingbox_refinement(V, ngpu=None, batchsize=1, minlength=5, dt=3, checkpoint=False):
    # Proposals:  Improve collector proposal for each video with an optimal object proposal.  This will result in filtering away a small number of hard positives.
    print('[pycollector.dataset]: bounding box refinement for %d videos' % (len(V)))
    model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  # =8 (0046) =20 (0053)
    V_dist = [(v,dt) for v in V]  
    B = vipy.batch.Batch(V_dist, ngpu=ngpu, strict=False, checkpoint=checkpoint)
    V = B.scattermap(lambda net,x: net(x[0], proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=x[1], mincover=0.8, byclass=True, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True).print(), model).result()  
    V = [v for v in V if (v is not None) and (not v.hasattribute('unrefined'))]  # remove videos that failed refinement
    V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
    if ngpu is not None:
        B.shutdown()  # garbage collect GPU resources
    return V


def split_dataset(A, trainfraction=0.7, testfraction=0.1, valfraction=0.2, seed=42, verbose=True):
    assert all([isinstance(a, vipy.video.Video) for a in A]), "Invalid input"
    
    np.random.seed(seed)
    d = vipy.util.groupbyasdict(A, lambda a: a.category())
    (trainset, testset, valset) = ([],[],[])
    for (c,v) in d.items():
        np.random.shuffle(v)
        if len(v) < 3:
            print('[pycollector.dataset]: skipping category "%s" with too few examples' % c)
            continue

        (testlist, vallist, trainlist) = vipy.util.dividelist(v, (testfraction, valfraction, trainfraction))
        testset.extend(testlist)
        valset.extend(vallist)  
        trainset.extend(trainlist)

    if verbose:
        print('[pycollector.dataset]: trainset=%d (%1.1f)' % (len(trainset), trainfraction))
        print('[pycollector.dataset]: valset=%d (%1.1f)' % (len(valset), valfraction))
        print('[pycollector.dataset]: testset=%d (%1.1f)' % (len(testset), testfraction))
    return (trainset, testset, valset)


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



def tohtml(outfile, pklfile=None, videolist=None, mindim=512, title='Visualization', fraction=1.0, display=False):
    """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
    assert pklfile is not None or videolist is not None, "Must provide either dataset pklfile or videolist=vipy.util.load(pklfile)"
    assert videolist is None or all([isinstance(v, vipy.video.Video) for v in videolist]), "videolist must be vipy.video.Video"
    assert ishtml(outfile), "Output file must be .html"
    assert pklfile is None or ispkl(pklfile), "Pickle file must be .pkl as part of a collector dataset"
    assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"

    import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
    import vipy.batch  # requires pip install vipy[all]
    dataset = vipy.util.load(pklfile) if pklfile is not None else videolist
    dataset = dataset if not isinstance(dataset, tuple) else dataset[0]  # backwards compatible
    dataset = [dataset[k] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]

    vipy.globals.max_workers(pct=0.8)
    quicklist = vipy.batch.Batch(dataset).filter(lambda v: not v.isdegenerate()).map(lambda v: (v.load().quicklook(), v.flush().print()))
    quicklooks = [imq for (imq, v) in quicklist]  # for HTML display purposes
    provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
    (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
    return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)


def activitymontage(pklfile, gridrows=30, gridcols=50, mindim=64):
    """30x50 activity montage, each 64x64 elements using the output of prepare_dataset"""
    (vidlist, d_category) = vipy.util.load(pklfile)
    actlist = [v.mindim(mindim) for v in vidlist]
    np.random.seed(42); random.shuffle(actlist)
    actlist = actlist[0:gridrows*gridcols]
    return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols).saveas(toextension(pklfile, 'mp4')).filename()


def activitymontage_bycategory(pklfile, gridcols=49, mindim=64):
    """num_categoryes x gridcols activity montage, each row is a category"""
    np.random.seed(42)
    (vidlist, d_category) = vipy.util.load(pklfile)

    actlist = []
    for k in sorted(d_category.keys()):
        actlist_k = [v for v in vidlist if v.category() == k]
        random.shuffle(actlist_k)
        assert len(actlist_k) >= gridcols
        actlist.extend(actlist_k[0:gridcols])
        outfile = os.path.join(filepath(pklfile), '%s_%s.mp4' % (filebase(pklfile), k))
        print(vipy.visualize.videomontage(actlist_k[0:15], 256, 256, gridrows=3, gridcols=5).saveas(outfile).filename())

    outfile = os.path.join(filepath(pklfile), '%s_%d_bycategory.mp4' % (filebase(pklfile), gridcols))
    print('[pycollector.dataset.activitymontage_bycategory]: rows=%s' % str(sorted(d_category.keys())))
    return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=len(d_category), gridcols=gridcols).saveas(outfile).filename()


class Dataset(object):
    def __init__(self,
                 indir,
                 format='.json'):

        indir = os.path.abspath(os.path.expanduser(indir))                
        
        self._indir = indir
        self._savefile_ext = format

        assert self._savefile_ext in ['.pkl', '.json'], "invalid format '%s'" % str(format)

    def __repr__(self):
        return str('<pycollector.dataset: "%s">' % self._indir)

    def isdataset(self):
        return len(self.transforms()) > 0

    def transform(self, src, dst, f_transform):        
        assert self.hastransform(src), "Source dataset '%s' does not exist" % self.savefile(src)                
        assert src != dst, "Source and destination cannot be the same"
        self.save(f_transform(self.load(src)), dst)
        return self
        
    def transforms(self):
        return [filebase(f) for f in listext(self._indir, self._savefile_ext)]
        
    def hastransform(self, transform):
        return transform in self.transforms()

    def save(self, videolist, transform, nourl=True, castas=vipy.video.Scene, relpath=True):
        if relpath:
            print('[pycollector.dataset]: setting relative paths')
            videolist = [v.relpath(filepath(self.savefile(transform))) for v in videolist]
        if nourl: 
            print('[pycollector.dataset]: removing URLs')
            videolist = [v.nourl() for v in videolist]           
        if castas is not None:
            assert hasattr(castas, 'cast'), "Invalid cast"
            print('[pycollector.dataset]: casting as "%s"' % (str(type(castas))))
            videolist = [castas.cast(v) for v in videolist]                     
        print('[pycollector.dataset]: Saving %d videos to "%s"' % (len(videolist), self.savefile(transform)))
        vipy.util.save(videolist, self.savefile(transform))

    def savefile(self, transform):
        return os.path.join(self._indir('%s%s' % (transform, self._savefile_ext)))
    
    def load(self, transform):
        assert self.hastransform(transform)
        print('[pycollector.dataset]: Loading "%s" ...' % self.savefile(transform))
        return vipy.util.load(self.savefile(transform))

    def tgz(self, tf, transformlist=[], extralist=[], bz2=False):
        assert (vipy.util.istgz(tf) and not bz2) or (vipy.util.isbz2(tf) and bz2)
        tf = os.path.abspath(os.path.expanduser(tf))       
        self.backup(tf)        

        assert all([self.hastransform(f) for f in transformlist]), "Invalid transform list '%s'" % str(transformlist)
        assert all([os.path.exists(f) for f in extralist]), "Invalid extra list '%s'" % str(extralist)
            
        filelist = [self.savefile(f) for f in transformlist if self.hastransform(f)] 
        filelist = filelist + [i for i in extralist if os.path.exists(i)]
        filelist = [f.replace(filepath(self._indir), '')[1:] for f in filelist]
        
        cmd = 'tar %scvf %s -C %s %s' % ('j' if bz2 else 'z', tf, filepath(self._indir), ' '.join(filelist))
        assert shutil.which('tar') is not None, "tar not found on path"
        print('[pycollector.dataset]: executing "%s"' % cmd)        
        os.system(cmd)

        # Too slow:
        # with tarfile.open(tf, mode=mode) as obj:
        #    for f in filelist:
        #        arcname = '%s/%s' % (filetail(self._indir), filetail(f))
        #        print('[pycollector.dataset]: %s -> %s' % (f, arcname))
        #        obj.add(f, arcname=arcname)

        return tf

    def md5sum(self, filename):
        """Equivalent to os.system('md5sum %s' % filename)"""
        assert isbz2(filename) or istgz(filename)
        return vipy.downloader.generate_md5(filename)
    
    def bz2(self, bz2file, transformlist=[], extralist=[]):
        assert vipy.util.isbz2(bz2file)
        return self.tgz(bz2file, transformlist, extralist, bz2=True)
        
    def split(self, transform):
        assert self.hastransform(transform)
        (trainset, valset, testset) = split_dataset(self.load(transform))

    def trainset(self):
        return self.load('trainset')

    def traincsv(self, csvfile=None):
        csv = [v.csv() for v in self.trainset()]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])

    def trainhtml(self, htmlfile=None, fraction=0.01, display=True):
        htmlfile = htmlfile if htmlfile is not None else temphtml()
        return tohtml(htmlfile, pklfile=self._trainpkl, mindim=512, title='Training set Visualization', fraction=fraction, display=display)
        
    def valset(self):
        return self.load('valset')        

    def valcsv(self, csvfile=None):
        csv = [v.csv() for v in self.valset()]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])
    
    def testset(self):
        return self.load('testset')                

    def testcsv(self, csvfile=None):
        csv = [v.csv() for v in self.testset()]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])
    

        
    
class ActivityDataset(Dataset):
    def videos_to_clips(self):
        assert os.path.exists(self._videos_pkl)        
        return vipy.util.save(activityclip(vipy.util.load(self._videos_pkl), remkdir(self._clips_dir)), self._clips_pkl)

    
class FaceDataset(Dataset):
    def videos_to_clips(self):
        assert os.path.exists(self._videos_pkl)        
        return vipy.util.save(subjectclip(vipy.util.load(self._videos_pkl), remkdir(self._clips_dir)), self._clips_pkl)

    
class ObjectDataset(Dataset):
    def videos_to_clips(self):
        assert os.path.exists(self._videos_pkl)        
        return vipy.util.save(objectclip(vipy.util.load(self._videos_pkl), remkdir(self._clips_dir)), self._clips_pkl)
    
