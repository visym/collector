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
import json

    
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


class TorchTensordir(torch.utils.data.Dataset):
    """A torch dataset stored as a directory of .pkl.bz2 files each containing a list of [(tensor, str=json.dumps(label)), ...] tuples used for data augmented training.
    
       This is useful to use the default Dataset loaders in Torch.
    
    .. note:: Use python random() and not numpy random 
    """
    def __init__(self, tensordir, verbose=True, reseed=True):
        assert os.path.isdir(tensordir)
        self._dirlist = [s for s in vipy.util.extlist(tensordir, '.pkl.bz2')]
        self._verbose = verbose
        self._reseed = reseed

    def __getitem__(self, k):
        if self._reseed:
            random.seed()  # force randomness after fork()

        assert k >= 0 and k < len(self._dirlist)
        for j in range(0,3):
            try:
                obj = vipy.util.bz2pkl(self._dirlist[k])  # load me
                assert len(obj) > 0, "Invalid augmentation"
                (t, lbl) = obj[random.randint(0, len(obj))]  # choose one tensor at random
                assert t is not None and json.loads(lbl) is not None, "Invalid augmentation"  # get another one if the augmentation was invalid
                return (t, lbl)
            except:
                time.sleep(1)  # try again after a bit if another process is augmenting this .pkl.bz2 in parallel
        if self._verbose:
            print('[pycollector.dataset.TorchTensordir][WARNING]: %s corrupted or invalid' % self._dirlist[k])
        return self.__getitem__(random.randint(0, len(self)))  # maximum retries reached, get another one

    def __len__(self):
        return len(self._dirlist)

    def filter(self, f):
        self._dirlist = [x for x in self._dirlist if f(x)]
        return self

class Dataset():
    """pycollector.dataset.Dataset() class
    
       This class is designed to be used with vipy.batch.Batch() for massively parallel operations 
    """

    def __init__(self, objlist_or_filename, id=None, abspath=True):
        objlist = vipy.util.load(objlist_or_filename, abspath=abspath) if (vipy.util.isjsonfile(objlist_or_filename) or vipy.util.ispklfile(objlist_or_filename)) else objlist_or_filename
        assert isinstance(objlist, list), "Invalid input"
        self._saveas_ext = ['pkl', 'json']
        self._id = id if id is not None else (vipy.util.filetail(objlist_or_filename) if isinstance(objlist_or_filename, str) else uuid.uuid4().hex)
        self._objlist = tolist(objlist)
        assert len(self._objlist) > 0, "Invalid object list"

    def __repr__(self):
        if len(self) > 0:
            return str('<pycollector.dataset: id="%s", len=%d, type=%s>' % (self.id(), len(self), str(type(self._objlist[0]))))
        else:
            return str('<pycollector.dataset: id="%s", len=0>' % (self.id()))

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
    def tolist(self):
        return self._objlist

    def flatten(self):
        self._objlist = [o for objlist in self._objlist for o in vipy.util.tolist(objlist)]
        return self

    def istype(self, validtype):
        return all([any([isinstance(v,t) for t in tolist(validtype)]) for v in self._objlist]), "invalid type - must be %s" % str(validtype)            
            
    def isvipy(self):
        return self.istype([vipy.image.Image, vipy.video.Video])

    def is_vipy_video(self):
        return self.istype([vipy.video.Video])

    def is_vipy_scene(self):
        return self.istype([vipy.video.Scene])

    def clone(self):
        return copy.deepcopy(self)

    def archive(self, tarfile, delprefix, mediadir='', format='json', castas=vipy.video.Scene, verbose=False, extrafiles=None, novideos=False):
        """Create a archive file for this dataset.  This will be archived as:

           /path/to/tarfile.{tar.gz|.tgz|.bz2}
              tarfilename
                 tarfilename.{json|pkl}
                 mediadir/
                     video.mp4
                 extras1.ext
                 extras2.ext
        
            Inputs:
              - tarfile: /path/to/tarfilename.tar.gz
              - delprefix:  the absolute file path contained in the media filenames to be removed.  If a video has a delprefix='/a/b' then videos with path /a/b/c/d.mp4' -> 'c/d.mp4', and {JSON|PKL} will be saved with relative paths to mediadir
              - mediadir:  the subdirectory name of the media to be contained in the archive.  Usually "videos".             
              - extrafiles: list of tuples [(abspath, filename_in_archive),...]

            Example:  

              - Input files contain /path/to/oldvideos/category/video.mp4
              - Output will contain relative paths videos/category/video.mp4

              >>> d.archive('out.tar.gz', delprefix='/path/to/oldvideos', mediadir='videos')
        
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
        D._objlist = [v.filename(v.filename().replace(os.path.normpath(delprefix), os.path.normpath(os.path.join(stagedir, mediadir))), symlink=not novideos) for v in D.list()]
        pklfile = os.path.join(stagedir, '%s.%s' % (filetail(filefull(tarfile)), format))
        D.save(pklfile, relpath=True, nourl=True, noadmin=True, castas=castas, significant_digits=2, noemail=True, flush=True)
    
        # Copy extras (symlinked) to staging directory
        if extrafiles is not None:
            for (e, a) in tolist(extrafiles):
                assert os.path.exists(os.path.abspath(e)), "Invalid extras file '%s'" % e
                os.symlink(os.path.abspath(e), os.path.join(stagedir, filetail(e) if a is None else a))

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

    def classes(self):
        return self.classlist()
    def categories(self):
        return self.classlist()

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
        
    def has(self, val, key):
        return any([key(obj) == val for obj in self._objlist])

    def replace(self, other, key):
        """Replace elements in self with other with equality detemrined by the key lambda function"""
        assert isinstance(other, Dataset), "invalid input"
        d = {key(v):v for v in other}
        self._objlist = [v if key(v) not in d else d[key(v)] for v in self._objlist]
        return self

    def merge(self, other, outdir, selfdir, otherdir):
        assert isinstance(other, Dataset), "invalid input"
        (selfdir, otherdir, outdir) = (os.path.normpath(selfdir), os.path.normpath(otherdir), vipy.util.remkdir(os.path.normpath(outdir)))
        assert all([selfdir in v.filename() for v in self._objlist])
        assert all([otherdir in v.filename() for v in other._objlist])

        D1 = self.clone().localmap(lambda v: v.filename(v.filename().replace(selfdir, outdir), copy=False, symlink=True))
        D2 = other.clone().localmap(lambda v: v.filename(v.filename().replace(otherdir, outdir), copy=False, symlink=True))
        return D1.union(D2)

    def augment(self, f, n_augmentations):
        assert n_augmentations >= 1
        self._objlist = [f(v.clone()) for v in self._objlist for k in range(n_augmentations)]  # This will remove the originals
        return self

    def filter(self, f):
        self._objlist = [v for v in self._objlist if f(v)]
        return self

    def valid(self):
        return self.filter(lambda v: v is not None)

    def takefilter(self, f, n=1):
        """Apply the lambda function f and return n elements in a list where the filter returns true
        
        Args:
            f: [lambda] If f(x) returns true, then keep
            n: [int >= 0] The number of elements to take
        
        Returns:
            [n=0] Returns empty list
            [n=1] Returns singleton element
            [n>1] Returns list of elements of at most n such that each element f(x) is True            
        """
        objlist = [obj for obj in self._objlist if f(obj)]
        return [] if (len(objlist) == 0 or n == 0) else (objlist[0] if n==1 else objlist[0:n])

    def to_jsondir(self, outdir):
        print('[pycollector.dataset]: exporting %d json files to "%s"...' % (len(self), outdir))
        vipy.util.remkdir(outdir)  # to avoid race condition
        Batch(vipy.util.chunklist([(k,v) for (k,v) in enumerate(self._objlist)], 64), as_completed=True, minscatter=1).map(lambda X: [vipy.util.save(x[1].clone(), os.path.join(outdir, '%s_%d.json' % (x[1].clone().videoid(), x[0]))) for x in X]).result()
        return outdir

    def takelist(self, n, category=None, canload=False):
        assert n >= 0, "Invalid length"

        outlist = []
        objlist = self._objlist if category is None else [v for v in self._objlist if v.category() == category]
        for k in np.random.permutation(range(0, len(objlist))).tolist():
            if not canload or objlist[k].isloadable():
                outlist.append(objlist[k])  # without replacement
            if len(outlist) == n:
                break
        return outlist

    def take(self, n, category=None, canload=False):
        return Dataset(self.takelist(n, category=category, canload=canload))

    def take_per_category(self, n, id=None, canload=False):
        return Dataset([v for c in self.categories() for v in self.takelist(n, category=c, canload=canload)], id=id)
    
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
        """Distributed map.

        To perform this in parallel across four processes:

        >>> with vipy.globals.parallel(4):
        >>>     self.map(lambda v: ...)

        """
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

    def frequency(self):
        return self.count()

    def percentage(self):
        """Fraction of dataset for each label"""
        d = self.count()
        n = sum(d.values())
        return {k:v/float(n) for (k,v) in d.items()}

    def multilabel_inverse_frequency_weight(self):
        """Return an inverse frequency weight for multilabel activities, where label counts are the fractional label likelihood within a clip"""
        assert self.is_vipy_video()

        lbl_likelihood  = {k:0 for k in self.classlist()}
        for v in self.list():
            if len(v.activities()) > 0:
                (ef, sf) = (max([a.endframe() for a in v.activitylist()]), min([a.startframe() for a in v.activitylist()]))  # clip length 
                lbl_frequency = vipy.util.countby([a for A in v.activitylabel(sf, ef) for a in A], lambda x: x)  # frequency within clip
                for (k,f) in lbl_frequency.items():
                    lbl_likelihood[k] += f/(ef-sf)

        # Inverse frequency weight on label likelihood per clip
        d = {k:1.0/max(v,1) for (k,v) in lbl_likelihood.items()}
        n = sum(d.values())  
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def inverse_frequency_weight(self):
        """Return inverse frequency weight for categories in dataset.  Useful for unbalanced class weighting during training"""
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
        assert self.isvipy()
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

    def to_torch(self, f_video_to_tensor):
        """Return a torch dataset that will apply the lambda function f_video_to_tensor to each element in the dataset on demand"""
        return TorchDataset(f_video_to_tensor, self)

    def to_torch_tensordir(self, f_video_to_tensor, outdir, n_augmentations=20, n_chunks=512):
        """Return a TorchTensordir dataset that will load a pkl.bz2 file that contains one of n_augmentations (tensor, label) pairs.
        
        This is useful for fast loading of datasets that contain many videos.

        """
        assert self.is_vipy_scene()
        outdir = vipy.util.remkdir(outdir)
        B = vipy.util.chunklist(self._objlist, n_chunks)
        vipy.batch.Batch(B, as_completed=True, minscatter=1).map(lambda V, f=f_video_to_tensor, outdir=outdir, n_augmentations=n_augmentations: [vipy.util.bz2pkl(os.path.join(outdir, '%s.pkl.bz2' % v.instanceid()), [f(v.clone()) for k in range(0, n_augmentations)]) for v in V])
        return TorchTensordir(outdir)

    def annotate(self, outdir, mindim=512):
        assert self.isvipy()
        f = lambda v, outdir=outdir, mindim=mindim: v.mindim(mindim).annotate(outfile=os.path.join(outdir, '%s.mp4' % v.videoid())).print()
        return self.map(f, dst='annotate')

    def tohtml(self, outfile, mindim=512, title='Visualization', fraction=1.0, display=False, clip=True, activities=True, category=True):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
        assert ishtml(outfile), "Output file must be .html"
        assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"
        
        import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
        import vipy.batch  # requires pip install vipy[all]

        dataset = self.list()
        assert all([isinstance(v, vipy.video.Video) for v in dataset])
        dataset = [dataset[k] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
        #dataset = [v for v in dataset if all([len(a) < 15*v.framerate() for a in v.activitylist()])]  # remove extremely long videos

        quicklist = vipy.batch.Batch(dataset, strict=False, as_completed=True, minscatter=1).map(lambda v: (v.load().quicklook(), v.flush().print())).result()
        quicklist = [x for x in quicklist if x is not None]  # remove errors
        quicklooks = [imq for (imq, v) in quicklist]  # keep original video for HTML display purposes
        provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)


    def video_montage(self, outfile, gridrows=30, gridcols=50, mindim=64, bycategory=False, category=None, annotate=True, trackcrop=False, transpose=False, max_duration=None, framerate=30, fontsize=8):
        """30x50 activity montage, each 64x64 elements.

        Args:
            outfile: [str] The name of the outfile for the video.  Must have a valid video extension. 
            gridrows: [int, None]  The number of rows to include in the montage.  If None, infer from other args
            gridcols: [int] The number of columns in the montage
            mindim: [int] The square size of each video in the montage
            bycategory: [bool]  Make the video such that each row is a category 
            category: [str, list] Make the video so that every element is of category.  May be a list of more than one categories
            annotate: [bool] If true, include boxes and captions for objects and activities
            trackcrop: [bool] If true, center the video elements on the tracks with dilation factor 1.5
            transpose: [bool] If true, organize categories columnwise, but still return a montage of size (gridrows, gridcols)
            max_duration: [float] If not None, then set a maximum duration in seconds for elements in the video.  If None, then the max duration is the duration of the longest element.

        Returns:
            A clone of the dataset containing the selected videos for the montage, ordered rowwise in the montage

        .. notes::  
            - If a category does not contain the required number of elements for bycategory, it is removed prior to visualization
            - Elements are looped if they exit prior to the end of the longest video (or max_duration)
        """
        assert self.is_vipy_video()
        assert vipy.util.isvideo(outfile)
        assert gridrows is None or (isinstance(gridrows, int) and gridrows >= 1)
        assert gridcols is None or (isinstance(gridcols, int) and gridcols >= 1)
        assert isinstance(mindim, int) and mindim >= 1
        assert category is None or isinstance(category, str)

        D = self.clone()
        if bycategory:
            (num_categories, num_elements) = (gridrows, gridcols) if not transpose else (gridcols, gridrows)
            assert num_elements is not None
            requested_categories = sorted(D.classlist()) if (num_categories is None) else sorted(D.classlist())[0:num_categories]             
            categories = [c for c in requested_categories if D.count()[c] >= num_elements]  # filter those categories that do not have enough
            if set(categories) != set(requested_categories):
                warnings.warn('[pycollector.dataset.video_montage]: removing "%s" without at least %d examples' % (str(set(requested_categories).difference(set(categories))), num_elements))
            vidlist = sorted(D.filter(lambda v: v.category() in categories).take_per_category(num_elements, canload=True).tolist(), key=lambda v: v.category())
            vidlist = vidlist if not transpose else [vidlist[k] for k in np.array(range(0, len(vidlist))).reshape( (len(categories), num_elements) ).transpose().flatten().tolist()] 
            (gridrows, gridcols) = (len(categories), num_elements) if not transpose else (num_elements, len(categories))
            assert len(vidlist) == gridrows*gridcols

        elif category is not None:
            vidlist = D.filter(lambda v: v.category() in vipy.util.tolist(category)).take(gridrows*gridcols, canload=True).tolist()            
        elif len(D) != gridrows*gridcols:
            vidlist = D.take(gridrows*gridcols, canload=True).tolist()
        else:
            vidlist = D.tolist()

        vidlist = [v.framerate(framerate) for v in vidlist]  # resample to common framerate (this may result in jittery tracks
        montage = Dataset(vidlist, id='video_montage').clone()  # for output
        vidlist = [v.trackcrop(dilate=1.5, maxsquare=True) if (v.trackbox() is not None) else v for v in vidlist] if trackcrop else vidlist  # may be None, if so return the video
        vidlist = [v.mindim(mindim) for v in vidlist]  # before annotate for common font size
        vidlist = [vipy.video.Video.cast(v) for v in vidlist] if not annotate else [v.annotate(verbose=False, fontsize=fontsize) for v in vidlist]  # pre-annotate
            
        vipy.visualize.videomontage(vidlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols, framerate=framerate, max_duration=max_duration).saveas(outfile)
        return montage


    def boundingbox_refinement(self, dst='boundingbox_refinement', batchsize=1, dt=3, minlength=5, f_savepkl=None):        
        """Must be connected to dask scheduler such that each machine has GPU resources"""
        model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize) 
        f = lambda net, v, dt=dt, f_savepkl=f_savepkl, b=batchsize: net.gpu(list(range(torch.cuda.device_count())), batchsize=b)(v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=dt, mincover=0.8, byclass=True, shapeiou=0.7, smoothing=None, strict=True).pklif(f_savepkl is not None, f_savepkl(v)).print()
        D = self.map(f, dst=dst, model=model)
        D.filter(lambda v: (v is not None) and (not v.hasattribute('unrefined')))  # remove videos that failed refinement
        D.localmap(lambda v: v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>minlength and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])))  # get rid of activities without tracks greater than dt
        return D

    def stabilize(self, f_saveas, dst='stabilize', padwidthfrac=1.0, padheightfrac=0.2):
        from vipy.flow import Flow
        f_stabilize = (lambda v, f_saveas=f_saveas, padwidthfrac=padwidthfrac, padheightfrac=padheightfrac: 
                       Flow(flowdim=256).stabilize(v, strict=False, residual=True, padwidthfrac=padwidthfrac, padheightfrac=padheightfrac, outfile=f_saveas(v)).pkl().print() if v.canload() else None)
        D = self.map(f_stabilize, dst=dst)
        D.filter(lambda v: (v is not None) and (not v.hasattribute('unstabilized')))  # remove videos that failed
        return D
    
    def track(self, dst='tracked'):
        return self.map(f=lambda net, v: net.track(v), dst=dst, model=pycollector.detection.MultiscaleVideoTracker())
        
        
    def zip(self, other, sortkey=None):
        """Zip two datasets.  Equivalent to zip(self, other).

        >>> for (d1,d2) in D1.zip(D2, sortkey=lambda v: v.instanceid()):
        >>>     pass
        
        >>> for (d1, d2) in zip(D1, D2):
        >>>     pass

        Args:
            other: [`pycollector.dataset.Dataset`] 
            sortkey: [lambda] sort both datasets using the provided sortkey lambda.
        
        Returns:
            Generator for the tuple sequence ( (self[0], other[0]), (self[1], other[1]), ... )
        """ 
        assert isinstance(other, Dataset)
        assert len(self) == len(other)

        for (vi, vj) in zip(self.sort(sortkey), other.sort(sortkey)):
            yield (vi, vj)

    def sort(self, key):
        """Sort the dataset in-place using the sortkey lambda function"""
        if key is not None:
            self._objlist.sort(key=key)
        return self
                
