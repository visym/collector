import os
import numpy as np
import pycollector.detection
from pycollector.globals import print
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv
import random
import vipy
import vipy.util
import shutil
from vipy.batch import Batch
import uuid
from pathlib import PurePath
import warnings
import copy 
import atexit


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


class Dataset():
    """pycollector.dataset.Dataset() class
    
       This class is designed to be used with vipy.batch.Batch() for massively parallel operations 
    """

    def __init__(self, indir, strict=True, ext=['pkl'], checkpoint=True):

        indir = os.path.abspath(os.path.expanduser(indir))                
        
        self._indir = indir
        assert os.path.isdir(indir), "invalid input directory"
        
        self._savefile_ext = tolist(ext)

        self._schema = (lambda dstdir, v, k=None, indir=self._indir, ext=None: os.path.join(indir, dstdir, v.category(), 
                                                                                            ('%s%s.%s' % (filebase(v.filename()), 
                                                                                                          ('_%s' % str(k)) if k is not None else '', 
                                                                                                          (fileext(v.filename(), withdot=False) if ext is None else str(ext))))
                                                                                            if (k is not None or ext is not None) 
                                                                                            else filetail(v.filename())))
        
        self._dataset = {}  # cache
        self._Batch = vipy.batch.Batch  # Batch() API
        self._strict = strict
        self._checkpoint = checkpoint
        self._stagedir = remkdir(os.path.join(indir, '.archive'))

    def __repr__(self):
        return str('<pycollector.dataset: "%s">' % self._indir)

    def map(self, src, dst, f_transform, model=None, f_rebatch=None, strict=True, save=True):        
        assert self.has_dataset(src), "Source dataset '%s' does not exist" % src                
        assert src != dst, "Source and destination cannot be the same"

        V_dst = self.dataset(dst) if self.has_dataset(dst) else []
        V_src = self.dataset(src)
        V_src = f_rebatch(V_src) if f_rebatch is not None else V_src
        B = self._Batch(V_src, strict=False, as_completed=True, checkpoint=self._checkpoint)
        V = B.map(f_transform).result() if not model else B.scattermap(f_transform, model).result()
        if any([v is None for v in V]):
            print('pycollector.dataset][%s->%s]: %d failed' % (src, dst, len([v for v in V if v is None])))
        V_dst.extend([v for v in V if v is not None]) 
        self._dataset[dst] = V_dst
        return self.save(dst) if save else self

    def new(self, newset, dst):
        assert all([isinstance(v, vipy.video.Video) or isinstance(v, vipy.image.Image) for vl in newset for v in tolist(vl)]), "Invalid input"
        if self._strict:            
            assert all([v.category() is not None for vl in newset for v in tolist(vl)]), "Invalid category"
            assert all([v.url() is not None or v.filename() is not None for vl in newset for v in tolist(vl)])
        self._dataset[dst] = newset
        return self

    def dedupe(self, src, key):
        self._dataset[src] = list({key(v):v for v in self.dataset(src)}.values())
        return self
        
    def datasets(self):
        return list(set([filebase(f) for e in self._savefile_ext for f in listext(self._indir, '.%s' % e)]).union(set(self._dataset.keys())))
        
    def has_dataset(self, src):
        return src in self.datasets()

    def union(self, srclist, dst, key=None):
        assert isinstance(srclist, list)
        srcset = [v for src in srclist for v in self.dataset(src)]
        if key is not None:
            srcset = list({key(v):v for v in srcset}.values())
        else:
            pass
        self._dataset[dst] = srcset
        return self
    
    def difference(self, srclistA, srclistB, dst, key):
        srcsetA = self.union(tolist(srclistA), '_union1', key=key).dataset('_union1')
        srcsetB = self.union(tolist(srclistB), '_union2', key=key).dataset('_union2')
        assert key is not None, "Key is required"

        idset = set([key(v) for v in srcsetA]).difference([key(v) for v in srcsetB])            
        diffset = [v for v in srcsetA if key(v) in idset]
        self._dataset[dst] = diffset
        return self
        
    def filter(self, src, f):
        self.load(src)
        self._dataset[src] = [v for v in self.dataset(src) if f(v) is True]
        return self

    def isloaded(self, d):
        return d in self._dataset


    def restore(self, dst):
        self._dataset[dst] = self._Batch.checkpoint()
        return self

    def save(self, dst, nourl=False, castas=None, relpath=False, noadmin=False, strict=True):
        assert self.has_dataset(dst) and self.isloaded(dst)
        videolist = self.dataset(dst)
        [self._saveas(videolist, f, nourl, castas, relpath, noadmin, strict) for f in self._savefiles()]
        return self

    def _savefiles(self, dst, ext=None):
        assert ext is None or isinstance(ext, list), "Invalid input"
        return [self._savefile(dst, ext) for ext in (self._savefile_ext if ext is None else ext)]

    def _savefile(self, d, format='pkl'): 
        return os.path.join(self._indir, '%s.%s' % (d, format))

    def load(self, src, format='pkl', flush=False):
        assert self.has_dataset(src)
        if not self.isloaded(src) or flush:
            print('[pycollector.dataset]: Loading "%s" ...' % self._savefile(src, format=format))
            self._dataset[src] = vipy.util.load(self._savefile(src))
        return self

    def rename(self, src, dst):
        assert self.has_dataset(src)
        [os.symlink(s, d) for (s, d) in zip(self._savefiles(src), self._savefiles(dst))]
        return self

    def name(self):
        return filetail(self._indir)

    def take(self, src, n):
        assert self.has_dataset(src)
        self._dataset[src] = [self._dataset[src][k] for k in random.sample(range(0, len(self.dataset(src))), n)]
        return self

    def _saveas(self, V, outfile, nourl=False, castas=None, relpath=False, noadmin=False, strict=True):
        if relpath:
            print('[pycollector.dataset]: setting relative paths')
            videolist = [v.relpath(filepath(self._savefile(dst))) if os.path.isabs(v.filename()) else v for v in videolist]
        if nourl: 
            print('[pycollector.dataset]: removing URLs')
            videolist = [v.nourl() for v in videolist]           
        if noadmin:
            videolist = [v.delattribute('admin') for v in videolist]
        if castas is not None:
            assert hasattr(castas, 'cast'), "Invalid cast"
            print('[pycollector.dataset]: casting as "%s"' % (str(castas)))
            videolist = [castas.cast(v) for v in videolist]                     
        if strict:
            assert all([isinstance(v, vipy.video.Video) or isinstance(v, vipy.image.Image) for v in self.dataset(src)])
            assert all([not vipy.util.isstring(v) or not is_email_address(v) for vid in V for (k,v) in vid.attributes.items()]), "Email addresses cannot be in archives"
            assert all([not v.hasattribute('admin') for v in self.dataset(distsrc)]), "Admin query cannot be in archive"
        print('[pycollector.dataset]: Saving %d videos to "%s"' % (len(videolist), outfile))
        vipy.util.save(videolist, outfile)
        return self

        
    def stage(self, out, src=None, srcname=None, srcdir=None, strict=True):        
        stagedir = remkdir(os.path.join(self._stagedir, out))
        if src is not None and self.has_dataset(src):
            V = self.dataset(src)
            oldsrcdir = os.path.split(V[0])[0]    # /path/to/srcdir/$CATEGORY/$ID
            newsrcdir = srcdir if srcdir is not None else oldsrcdir
            V = [v.filename(os.path.relpath(v.filename(), filepath(self._indir)).replace(oldsrcdir, newsrcdir)) for v in V]
            self._saveas(V, os.path.join(stagedir, '%s.pkl' % (srcname)), relpath=True, nourl=True, castas=vipy.video.Scene, strict=strict)
            os.simlink(os.path.join(self._indir, oldsrcdir), os.path.join(stagedir, newsrcdir))

        elif src is not None and os.path.exists(src):
            os.simlink(src, os.path.join(stagedir, filetail(src) if srcname is None else srcname))
        else:
            remkdir(stagedir, flush=True)            

        return self

    def archive(self, out, outfile=None):  #, datasetlist, extras=None, root=None, ext=None, strict=True):

        stagedir = os.path.join(self._stagedir, out)
        assert os.path.isdir(stagedir)
        raise

        """Create a archive file from a list of datasets.  This will be archived as:

           outfile.{tar.gz|.tgz|.bz2}
               outfile
                 dataset1.{json|pkl}
                 dataset2.{json|pkl}
                 dataset1/
                     video3.mp4
                 dataset2/
                     video4.mp4
                 extras1.ext
                 extras2.ext
    
           -outfile [str]:  the filename of the archive.  The filebase will be used as the root of the archive file.
           -datasetlist [list]:  a list of datasets(), must be elements in datasets()
           -extras [list]:  a list of absolute paths to files that will be copied into the root
           
        """
        assert vipy.util.istgz(outfile) or vipy.util.isbz2(outfile), "Allowable extensions are .tar.gz, .tgz or .bz2"
        assert all([self.has_dataset(d) for d in datasetlist]), "Invalid dataset list '%s'" % str(datasetlist)
        root = filetail(outfile.replace('.tar.gz','').replace('.tgz','').replace('.bz2','')) if root is None else root
        if root != self.name():
            warnings.warn('Archive file "%s" with root "%s" does not match dataset name "%s"' % (outfile, root, self.name()))

        deletelist = []
        archivelist = []
        for src in datasetlist:
            assert all([isinstance(v, vipy.video.Video) or isinstance(v, vipy.image.Image) for v in self.dataset(src)])
            assert '_dist' not in src, "Archive must be created from original source dataset"
            distsrc = '%s_dist' % src  # relocatable, sanitized, sharable, will be renamed back to src in archive
            self.save(self.dataset(src), distsrc, relpath=True, nourl=True, castas=vipy.video.Scene)
            deletelist.extend(self._savefiles(distsrc))  # for cleanup            
            archivelist.extend([os.path.relpath(f, filepath(self._indir)) for f in self._savefiles(distsrc, ext=tolist(ext))])
            archivelist.extend([os.path.relpath(v.filename(), filepath(self._indir)) for v in self.dataset(distsrc)]) 

            if strict:
                assert all([not vipy.util.isstring(v) or not is_email_address(v) for vid in self.dataset(distsrc) for (k,v) in vid.attributes.items()]), "Email addresses cannot be in archives"
                assert all([not v.hasattribute('admin') for v in self.dataset(distsrc)]), "Admin query cannot be in archive"

        # Tar archive
        filelist = vipy.util.writelist(archivelist, tempcsv())
        outfile = os.path.abspath(os.path.expanduser(outfile))       
        cmd = ('tar %scvf %s %s %s -C %s --files-from %s ' % ('j' if vipy.util.isbz2(outfile) else 'z', 
                                                              outfile, 
                                                              '--transform "flags=r;s|%s/|%s/|"' % (self.name(), root),
                                                              ' '.join(['--transform "flags=r;s|%s|%s|"' % (filetail(self._savefile('%s_dist') % src), filetail(self._savefile(src))) for src in datasetlist]),
                                                              filepath(self._indir),
                                                              filelist))

        if extras is not None:
            assert isinstance(extras, list) and all([os.path.exists(f) for f in extras]), "Invalid extras"
            d = tempdir()
            args = ['-C', os.path.join(d)]
            for f in set(extras):
                shutil.copyfile(f, os.path.join(remkdir(os.path.join(d, self.name())), filetail(f)))
                deletelist.append(os.path.join(d, self.name(), filetail(f)))
                args.append(os.path.join(self.name(), filetail(f)))
            cmd += ' '.join(args)

        assert shutil.which('tar') is not None, "tar not found on path"
        print('[pycollector.dataset]: executing "%s"' % cmd)        
        print('[pycollector.dataset]: -Note that tar will print out the pre-transform filenames during add and not the extract filenames ...')
        os.system(cmd)  # too slow to use python "tarfile" package

        # Cleanup
        deletelist.append(filelist)
        for f in deletelist:
            if os.path.exists(f):
                os.remove(f) 

        print('[pycollector.dataset]: %s, MD5=%s' % (outfile, self.md5sum(outfile)))
        return outfile

    def md5sum(self, filename):
        """Equivalent to os.system('md5sum %s' % filename)"""
        assert isbz2(filename) or istgz(filename)
        return vipy.downloader.generate_md5(filename)
    
    def split_like(src, likeset, dst, key):
        ids = set([key(v) for v in likeset])
        self.new([v for v in self.dataset(src) if key(v) in ids], dst).save(dst)
        return self

    def split(src, traindst, testdst, valdst, trainfraction=0.7, testfraction=0.1, valfraction=0.2, seed=42):
        A = self.dataset(src)
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

        print('[pycollector.dataset]: trainset=%d (%1.1f)' % (len(trainset), trainfraction))
        print('[pycollector.dataset]: valset=%d (%1.1f)' % (len(valset), valfraction))
        print('[pycollector.dataset]: testset=%d (%1.1f)' % (len(testset), testfraction))
        
        self._dataset[traindst] = trainset
        self._dataset[testdst] = testset
        self._dataset[valdst] = valset
        return self


    def tocsv(self, src, csvfile=None):
        csv = [v.csv() for v in self.dataset(src)]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])

    def dataset(self, src, flush=False):
        return self.load(src, flush=flush)._dataset[src]

    def fetch(self, src, dst):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_fetch = (lambda v, f=f_saveas: v.filename(f(v)).download().print())
        return self.map(src, dst, f_fetch)
        

    def stabilize_refine_activityclip(self, src, dst, batchsize=1, dt=5, minlength=5, maxsize=512*3):
        from vipy.flow import Flow
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  # =8 (0046) =20 (0053)
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
        return self.save(V, dst)

    def refine_activityclip(self, src, dst, batchsize=1, dt=5, minlength=5):
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  # =8 (0046) =20 (0053)
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
        return self.save(V, dst)

    def stabilize(self, src, dst):
        from vipy.flow import Flow
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_stabilize = (lambda v, f=f_saveas: Flow(flowdim=256).stabilize(v, strict=False, residual=True).saveas(f(v), flush=True).print() if v.canload() else None)
        return self.map(src, dst, f_stabilize)

    def refine(self, src, dst, batchsize=1, dt=3, minlength=5):        
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  # =8 (0046) =20 (0053)
        f = lambda net, v, dt=dt: net(v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=dt, mincover=0.8, byclass=True, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True).print()
        V = self.map(src, dst, f, model=model, save=False).dataset(dst)
        V = [v for v in V if (v is not None) and (not v.hasattribute('unrefined'))]  # remove videos that failed refinement
        V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
        return self.save(V, dst)
            
    def trackcrop(self, src, dst, dilate=1.0, mindim=512, maxsquare=True):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_trackcrop = lambda v, d=dilate, m=mindim, f=f_saveas, b=maxsquare: v.trackcrop(dilate=d).mindim(m).maxsquareif(b).saveas(f(v)).print() if v.hastracks() else None
        return self.map(src, dst, f_trackcrop)        

    def tubelet(self, src, dst, dilate=1.0, maxdim=512):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_tubelet = lambda v, d=dilate, m=maxdim, f=f_saveas: v.activitytube(dilate=d, maxdim=512).saveas(f(v)).print() if v.hastracks() else None
        return self.map(src, dst, f_tubelet)        

    def resize(self, src, dst, mindim):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f_tubelet = lambda v, m=mindim, f=f_saveas: v.mindim(m).saveas(f(v)).print()
        return self.map(src, dst, f_tubelet)        

    def tohtml(self, src, outfile, mindim=512, title='Visualization', fraction=1.0, display=False, clip=True, activities=True, category=True):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
        assert ishtml(outfile), "Output file must be .html"
        assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"
        
        import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
        import vipy.batch  # requires pip install vipy[all]

        dataset = self.dataset(src)
        assert all([isinstance(v, vipy.video.Video) for v in dataset])
        dataset = [dataset[k] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
        
        quicklist = self._Batch(dataset).map(lambda v: (v.load().quicklook(), v.flush().print()) if v.canload() else (None, None)).result()
        quicklooks = [imq for (imq, v) in quicklist if imq is not None]  # keep original video for HTML display purposes
        provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)

    def activitymontage(src, gridrows=30, gridcols=50, mindim=64, bycategory=False):
        """30x50 activity montage, each 64x64 elements using the output of prepare_dataset"""
        vidlist = self.dataset(src)
        actlist = [v.mindim(mindim) for v in vidlist]
        np.random.seed(42); random.shuffle(actlist)
        actlist = actlist[0:gridrows*gridcols]
        return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols).saveas(toextension(pklfile, 'mp4')).filename()

    def activitymontage_bycategory(src, gridcols=49, mindim=64):
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
            outfile = os.path.join(filepath(pklfile), '%s_%s.mp4' % (filebase(pklfile), k))
            print(vipy.visualize.videomontage(actlist_k[0:15], 256, 256, gridrows=3, gridcols=5).saveas(outfile).filename())

        outfile = os.path.join(filepath(pklfile), '%s_%d_bycategory.mp4' % (filebase(pklfile), gridcols))
        print('[pycollector.dataset.activitymontage_bycategory]: rows=%s' % str(sorted(categories)))
        return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=len(categories), gridcols=gridcols).saveas(outfile).filename()


    def copy(self, src, dst):
        f_saveas = lambda v, dstdir=dst, f=self._schema: f(dstdir, v)
        f = lambda v, f=f_saveas: v.saveas(f(v)).print() if v.isdownloaded() else v.download().saveas(f(v)).print()
        return self.map(src, dst, f)

    def activityclip(self, src, dst):
        pass

    def trackclip(self, src, dst):
        pass
        

    def trainset(self):
        return self.dataset('trainset')

    def valset(self):
        return self.dataset('valset')        

    def testset(self):
        return self.dataset('testset')                

    def restore(self, dst):
        """Restore destination from checkpoint"""
        pass
        
    
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

