import os
import random
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import copy
import decimal
from decimal import Decimal
import calendar
import pytz
import hashlib
import uuid
import urllib
import xmltodict  
from boto3.dynamodb.conditions import Key, Attr
import webbrowser

import vipy
assert vipy.version.is_at_least('1.8.24')
from vipy.util import readjson, isS3url, tempjson,tempdir, totempdir, remkdir
from vipy.util import flatlist, tolist, groupbyasdict, writecsv, filebase, filetail, filepath, fileext, isurl, tolist
from vipy.object import Track
import vipy.version
import vipy.activity
from vipy.video import Scene
from vipy.geometry import BoundingBox
import vipy.downloader
import vipy.version

from pycollector.util import allmondays_since, yyyymmdd_to_date, is_email_address, isday, is_more_recent_than, nextday, lastmonday
from pycollector.util import lowerif, timestamp, fromdate, ismonday
from pycollector.globals import print

try:
    import ujson as json  # faster
except ImportError:
    import json


class Video(Scene):
    """pycollector.video.Video class
    
    """

    def __init__(self,
                 mp4file=None,
                 mp4url=None,
                 jsonurl=None,
                 jsonfile=None,
                 mindim=512,
                 dt=1,
                 fetch=True,
                 attributes=None,
    ):
        assert (mp4file is not None or mp4url is not None), "Invalid input - Must provide either mp4file or mp4url"
        assert (jsonurl is not None or jsonfile is not None), "Invalid input - Must provide either jsonurl or jsonfile"
        assert mp4url is None or isS3url(mp4url), "Invalid input - mp4url must be of the form returned from pycollector.project"
        assert jsonurl is None or isS3url(jsonurl), "Invalid input - jsonurl must be of the form returned from pycollector.project"

        # AWS credentials (if needed) must be set by pycollector.user
        if ((jsonurl is not None and (jsonfile is None or not os.path.exists(jsonfile))) or
            (mp4url is not None and (mp4file is None or not os.path.exists(mp4file)))):
             assert 'VIPY_AWS_ACCESS_KEY_ID' in os.environ and 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ, "AWS access keys not found - Log in using pycollector.user"
        
        # Vipy video constructor
        super().__init__(url=mp4url, filename=mp4file, attributes=attributes)

        # Video attributes
        #self._quicklook_url = "https://%s.s3.amazonaws.com/Quicklooks/%%s_quicklook_%%s_%%d.jpg" % (backend().s3_bucket())   # FIXME: pycollector.admin.video
        self._mp4url = mp4url
        self._mp4file = mp4file
        self._jsonurl = jsonurl
        self._jsonfile = os.path.abspath(os.path.expanduser(jsonfile)) if jsonfile is not None else jsonfile
        self._dt = dt
        self._is_json_loaded = None
        self._mindim = mindim
        self._verbose = False  # FIXME
        if fetch:
            self._load_json()

    @classmethod
    def cast(self, v):
        assert isinstance(v, vipy.video.Scene), "Invalid input - must be derived from vipy.video.Scene"        
        v.__class__ = Video
        return v
    
    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        v = Scene.from_json(d)  
        v._is_json_loaded = d['_is_json_loaded']
        v._dt = d['_dt']
        v._mindim = d['_mindim']
        v._verbose = d['_verbose']
        v._jsonfile = d['_jsonfile']
        v._jsonurl = d['_jsonurl']        
        v._mp4file = d['_mp4file']
        v._mp4url = d['_mp4url']
        v.__class__ = Video
        return v
        
    def json(self, encode=True):
        d = super().json(encode=False)
        d['_is_json_loaded'] = self._is_json_loaded
        d['_dt'] = self._dt
        d['_mindim'] = self._mindim
        d['_verbose'] = self._verbose
        d['_jsonfile'] = self._jsonfile
        d['_jsonurl'] = self._jsonurl        
        d['_mp4file'] = self._mp4file
        d['_mp4url'] = self._mp4url
        return json.dumps(d) if encode else d
    
    def __repr__(self):
        return str("<pycollector.video: uploaded=%s, activities=%s, scene=%s>" % (self.timestamp().strftime("%Y-%m-%d %H:%M")
                                                                                  if self.timestamp() is not None
                                                                                  else str(None),
                                                                                  self.activity_categories(),
                                                                                  str(super().__repr__())))
    
    def _load_json(self):
        """Lazy JSON download, parse, and import"""

        # Already loaded?  Call once
        if self._is_json_loaded is not None:
            return self

        # Not downloaded?
        if not self.hasjson():
            self.fetchjson()

        # Parse JSON (with version error handling)
        jsonfile = self._jsonfile
        if jsonfile is not None and os.path.getsize(jsonfile) != 0:
            if self._verbose:
                print('[pycollector.video]:  Parsing "%s"' % jsonfile)

            d = readjson(jsonfile)
            if "collection_id" not in d["metadata"]:
                d["metadata"]["collection_id"] = d["metadata"]["video_id"]  # android 1.1.1(3) bug

            for obj in d["object"]:
                if "label" not in obj:
                    obj["label"] = "person"  # android 1.1.1(3) bug
                if "label" in obj and obj["label"] == "vehicle":
                    # obj['label'] = 'person'  # all bug
                    pass
                for bb in obj["bounding_box"]:
                    if "frame_index" not in bb and "frameIndex" in bb:
                        bb["frame_index"] = bb["frameIndex"]  # android 1.1.1(3) bug

            d["metadata"]["rotate"] = None
            if d["metadata"]["orientation"] == "landscape":
                # d['metadata']['rotate'] = 'rot90cw'
                pass
            elif d["metadata"]["orientation"] == "portrait":
                # d['metadata']['rotate'] = 'rot90ccw'
                pass
            else:
                pass

            if "device_type" in d["metadata"] and "device_identifier" == "ios":
                d["metadata"]["rotate"] = "rot90ccw"  # iOS (7) bug, different than iOS (6)

            # FIXME: "collected_date":"2020-06-19T18:34:33+0000" on both now
            try:
                uploaded = datetime.strptime(
                    d["metadata"]["collected_date"], "%Y-%m-%d %H:%M:%S %z"
                )  # iOS 1.0 (6)
            except:
                try:
                    uploaded = datetime.strptime(
                        d["metadata"]["collected_date"], "%Y-%m-%d %I:%M:%S %p %z"
                    )  # bug number 55
                except:
                    uploaded = datetime.strptime(
                        d["metadata"]["collected_date"], "%Y-%m-%dT%H:%M:%S%z"
                    )  # android 1.1.1 (3)

            #if isapi('v1'):
            #    d["metadata"]["collected_date"] = uploaded.strftime(
            #        "%Y-%m-%d %H:%M:%S"
            #    )
            #else:
            #    et = pytz.timezone("US/Eastern")                
            #    d["metadata"]["collected_date"] = uploaded.astimezone(et).strftime("%Y-%m-%d %H:%M:%S")
            
            et = pytz.timezone("US/Eastern")                
            d["metadata"]["collected_date"] = uploaded.astimezone(et).strftime("%Y-%m-%d %H:%M:%S")
                

        else:
            print('[pycollector.video]: empty JSON "%s"' % jsonfile)
            d = None

        # Backwards compatible video import: should not be necessary with new app release
        if d is not None and not 'category' in d['metadata']:
            vipy.util.try_import('pycollector.admin.globals', message="Not authorized - Old style JSON requires admin access")
            from pycollector.admin.globals import backend, isapi
            from pycollector.admin.legacy import applabel_to_longlabel, shortname_synonyms, applabel_to_piplabel

            # V1 - old collection name pattern
            if any([d["metadata"]["collection_id"] in k for k in applabel_to_piplabel().keys()]):
                try:
                    d['metadata']['collection_name'] = d["metadata"]["collection_id"]
                    applabel = ['%s_%s_%s' % (d["metadata"]["project_id"], d["metadata"]["collection_id"], a['label']) for a in d['activity']]
                    applabel = [a if a in applabel_to_piplabel() else '%s_%s_%s' % (d["metadata"]["project_id"], d["metadata"]["collection_id"], shortname_synonyms()[a.split('_')[2]]) for a in applabel]
                    d['metadata']['category'] = ','.join([applabel_to_piplabel()[a] for a in applabel])
                    d['metadata']['shortname'] = ','.join([a.split('_')[2] for a in applabel])
                except Exception as e:
                    print('[pycollector.video]: legacy json import failed for v1 JSON "%s" with error "%s"' % (str(d['metadata']), str(e)))
                    d = None

            # V2 - new collection names, but activity names not in JSON
            elif isapi('v1') or isapi('v2'):
                version = 'v1' if isapi('v1') else 'v2'
                if version == 'v1':
                    backend('prod', 'v2')  # temporary switch

                if not backend().collections().iscollectionid(d["metadata"]["collection_id"]):
                    print('[pycollector.video]: invalid collection ID "%s"' % d["metadata"]["collection_id"])
                    d = None
                elif len(d['activity']) == 1 and len(d['activity'][0]['label']) == 0:
                    d['activity'] = []
                    d['metadata']['category'] = ''
                    d['metadata']['shortname'] = ''                    
                else:
                    try:
                        # Fetch labels from backend (with legacy shortname translation)
                        C = backend().collections()[d["metadata"]["collection_id"]]
                        d['metadata']['collection_name'] = backend().collections().id_to_name(d["metadata"]["collection_id"])
                        shortnames = []
                        for a in d['activity']:
                            if not (a['label'] in C.shortnames() or a['label'] in shortname_synonyms()):
                                raise ValueError("Invalid shortname '%s' for collection shortnames '%s' and not in legacy synonyms '%s'" % (a['label'], str(C.shortnames()), str(shortname_synonyms())))
                            shortnames.append(a['label'] if a['label'] in C.shortnames() else shortname_synonyms()[a['label']])
                        d['metadata']['category'] = ','.join([C.shortname_to_activity(s, strict=False) for s in shortnames])
                        d['metadata']['shortname'] = ','.join([s for s in shortnames])
                    except Exception as e: 
                        print('[pycollector.video]: label fetch failed for %s with exception %s' % (str(d['activity']), str(e)))
                        d = None

                if version == 'v1':
                    backend('prod', 'v1')  # switch back
            else:
                print('[pycollector.video]: Legacy JSON import failed for JSON with metadata - "%s"' % str(d["metadata"]))
                d = None
            
        else:
            # New style JSON: use labels stored directly in JSON
            pass
        
        # Import JSON into scene
        if d is not None:

            # TODO - Replace with video_data
            collection_name = d['metadata']['collection_name']
            
            self.category(collection_name)
            self.attributes = {} if self.attributes is None else self.attributes
            self.attributes.update(d['metadata'])
            self.framerate(int(round(float(d["metadata"]["frame_rate"]))))

            # FIXME: this videoID '20200421_1500081666724286' has low framerate.  Parsing is correct, but load() and show() is too fast
            # This requires explicitly setting output framerate in vipy.video

            # Import tracks
            d_trackid_to_track = {}
            for obj in d["object"]:

                keyboxes = [
                    BoundingBox(
                        xmin=bb["frame"]["x"],
                        ymin=bb["frame"]["y"],
                        width=bb["frame"]["width"],
                        height=bb["frame"]["height"],
                    )
                    for bb in sorted(
                        obj["bounding_box"], key=lambda x: x["frame_index"]
                    )
                ]
                keyframes = [
                    bb["frame_index"]
                    for bb in sorted(
                        obj["bounding_box"], key=lambda x: x["frame_index"]
                    )
                ]

                badboxes = [bb for bb in keyboxes if not bb.isvalid()]
                if len(badboxes) > 0:
                    print('[pycollector.video]: Removing %d bad keyboxes "%s" for videoid=%s' % (len(badboxes), str(badboxes), d["metadata"]["video_id"]))
                if len(badboxes) == len(keyboxes):
                    raise ValueError("all keyboxes in track are invalid")

                t = Track(
                    category=obj["label"],
                    framerate=float(d["metadata"]["frame_rate"]),
                    keyframes=[int(f) for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()],
                    boxes=[bb for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()],
                    boundary="strict"
                )
                if vipy.version.is_at_least("0.8.3"):
                    self.add(t, rangecheck=False)  # no rangecheck since all tracks are guarnanteed to be within image rectangle
                else:
                    self.add(t)
                d_trackid_to_track[t.id()] = t

            
            # Category variants:  a_category_name#Variant1=A&Joint=a_joint_label:Short Label&Variant2=B
            variant = {}
            d_shortname_to_category = {s:c for (s,c) in zip(d['metadata']['shortname'].split(','), d['metadata']['category'].split(','))}            
            if '#' in d['metadata']['category']:
                d_shortname_to_category = {s:c.split('#')[0] for (s,c) in d_shortname_to_category.items()}  # shortname and category may be empty
                variantlist = list(set([c.split('#')[1] if '#' in c else None for c in d['metadata']['category'].split(',')]))
                if len(variantlist) != 1:
                    print('[pycollector.video]: WARNING - Ignoring mixed variant "%s"' % str(variantlist))
                elif all([len(v)==0 for v in variantlist]):
                    pass  # empty variant
                elif any(['=' not in v or v.count('&') != (v.count('=')-1) for v in variantlist]):
                    print('[pycollector.video]: WARNING - Ignoring invalid variant "%s"' % str(variantlist))                    
                else:
                    variant = {k.split('=')[0]:k.split('=')[1] for k in variantlist[0].split('&')}
            self.attributes['variant'] = variant

            # Import activities
            for a in d["activity"]:
                try:
                    # Legacy shortname display
                    if a['label'] not in d_shortname_to_category:
                        if a['label'] not in shortname_synonyms():
                            raise ValueError("Invalid shortname '%s' for collection shortnames '%s' and not in legacy synonyms '%s'" % (a['label'], d_shortname_to_category, str(shortname_synonyms())))
                        a['label'] = a['label'] if a['label'] in d_shortname_to_category else shortname_synonyms()[a['label']]  # legacy translation              
                    if (d["metadata"]["collection_id"] == "P004C009" and d["metadata"]["device_identifier"] == "android"):
                        shortlabel = "Buying (Machine)"
                    elif (d["metadata"]["collection_id"] == "P004C008" and d["metadata"]["device_identifier"] == "ios" and "Purchasing" in a["label"]):
                        # BUG: iOS (11) reports wrong collection id for "purchase something from a machine" as P004C008 instead of P004C009
                        shortlabel = "Buying (Machine)"
                    elif (d["metadata"]["collection_id"] == "P004C009" and d["metadata"]["device_identifier"] == "ios"):
                        # BUG: iOS (11) reports wrong collection id for "pickup and dropoff with bike messenger" as P004C009 instead of P004C010
                        shortlabel = a["label"]  # unchanged
                    elif d["metadata"]["collection_id"] == "P005C003":
                        shortlabel = "Buying (Cashier)"
                    else:
                        shortlabel = a["label"]

                    #category = backend().collection()[d["metadata"]["collection_id"]].shortname_to_activity(a["label"])
                    category = d_shortname_to_category[a['label']]
                    self.add(vipy.activity.Activity(category=category,
                                                    shortlabel=shortlabel,
                                                    startframe=int(a["start_frame"]),
                                                    endframe=int(a["end_frame"]),
                                                    tracks=d_trackid_to_track,
                                                    framerate=d["metadata"]["frame_rate"]))
                                            
                except Exception as e:
                    print(
                        '[pycollector.video]: Filtering invalid activity JSON "%s" with error "%s" for videoid=%s'
                        % (str(a), str(e), d["metadata"]["video_id"])
                    )

            # Joint activity?  Occurs simultaneously with any JSON defined activities
            if 'Joint' in variant:
                self.add(vipy.activity.Activity(category=variant['Joint'].split(':')[0],
                                                shortlabel=variant['Joint'].split(':')[1] if ':' in variant['Joint'] else None,
                                                startframe=min([int(a["start_frame"]) for a in d["activity"]]) if len(d["activity"])>0 else 0,
                                                endframe=max([int(a["end_frame"]) for a in d["activity"]]) if len(d["activity"])>0 else int(np.round(float(d["metadata"]["duration"])*float(d["metadata"]['frame_rate']))),
                                                tracks=d_trackid_to_track,
                                                framerate=d["metadata"]["frame_rate"]))


            if d["metadata"]["rotate"] == "rot90ccw":
                self.rot90ccw()
            elif d["metadata"]["rotate"] == "rot90cw":
                self.rot90cw()

            self._is_json_loaded = True

            # Minimum dimension of video for reasonably fast interactions (must happen after JSON load to get frame size from JSON)
            if self._mindim is not None:
                if "frame_width" in self.metadata() and "frame_height" in self.metadata():  # older JSON bug
                    s = float(min(int(self.metadata()["frame_width"]), int(self.metadata()["frame_height"])))
                    if s > 256:
                        self.rescale(self._mindim / float(s))  # does not require load
                    else:
                        print('[pycollector.video]: Filtering Invalid JSON (height, width)')
                        self._is_json_loaded = False
                else:
                    assert vipy.version.is_at_least("0.8.0")
                    self.clear()  # remove this old video from consideration
                    self._is_json_loaded = False
        else:
            print('[pycollector.video]: JSON load failed - SKIPPING')
            self._is_json_loaded = False

        # Resample tracks
        if self._dt > 1 and self._is_json_loaded:
            self.trackmap(lambda t: t.resample(self._dt).significant_digits(2))

        assert vipy.version.is_at_least('1.8.34')            
        self.trackmap(lambda t: t.significant_digits(2))            
        return self

    def isedited(self):
        return '_' in self._jsonfile and filebase(self._jsonfile).split('_')[0] == self.videoid()  # edited JSON has the structure $VIDEOID_TIMESTAMP.json

    def editedat(self):
        return filebase(self._jsonfile).split('_')[1] if self.isedited() else None

    def variant(self):
        """Category variant"""
        return self.attributes['variant'] if 'variant' in self.attributes else None
    
    def geolocation(self):
        assert 'ipAddress' in self.metadata(), "Invalid JSON"
        url = 'http://api.geoiplookup.net/?query=%s' % self.metadata()['ipAddress']
        with urllib.request.urlopen(url) as f:
            response = f.read().decode('utf-8')
        d = xmltodict.parse(response)
        return dict(d['ip']['results']['result'])  
        
    def fetch(self, ignoreErrors=False):
        """Download JSON and MP4 if not already downloaded"""
        self.fetchjson() # Do we need this?
        return self.fetchvideo()

    def fetchvideo(self, ignoreErrors=False):
        super().fetch()
        return self

    def fetchjson(self):
        """Download JSON if not already downloaded"""

        if self._jsonfile is None:
            self._jsonfile = os.path.join(
                remkdir(
                    os.environ["VIPY_CACHE"]
                    if "VIPY_CACHE" in os.environ
                    else tempdir()
                ),
                filetail(self._jsonurl),
            )
            if not os.path.exists(self._jsonfile):
                print('[pycollector.video]:  Fetching "%s"' % self._jsonurl)
                try:
                    vipy.downloader.s3(self._jsonurl, self._jsonfile) 

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(
                        '[pycollector.video]: S3 download error "%s" - SKIPPING'
                        % str(e)
                    )
                    jsonfile = None
        return self

    def is_json_loaded(self):
        return self._is_json_loaded

    def hasjson(self):
        self.fetchjson()
        return os.path.exists(self._jsonfile)

    def hasMP4(self):
        return self.fetch().hasfilename()


    def activity_categories(self):
        """Return a set of unique activity categories in the video, not including object categories"""
        self._load_json()
        return set([a.category() for a in self._load_json().activities().values()])

    
    def quicklooks(self, n=9, dilate=1.5, mindim=256, fontsize=10, context=True):
        """Return a vipy.image.Image object containing a montage quicklook for each of the activities in this video.  
        
        Usage:
         
        >>> filenames = [im.saveas('/path/to/quicklook.jpg') for im in self.quicklooks()]
        
        """
        assert vipy.version.is_at_least("0.8.2")
        print('[pycollector.video]: Generating quicklooks for video "%s"' % self.videoid())
        return [a.quicklook(n=n,
                            dilate=dilate,
                            mindim=mindim,
                            fontsize=fontsize,
                            context=context)
                for a in self.fetch().activityclip()]

    def trim(self, padframes=0):
        """Temporally clip the video so that the video start is the beginning of the first activity, and the end of the video is the end of the last activity.
        Optionally add a temporal pad of padframes before and after the clip"""
        startframe = max(0, min([a.startframe() for (k, a) in self.fetch().activities().items()]))
        endframe = max([a.endframe() for (k, a) in self.activities().items()])
        self.clip(startframe - padframes, endframe + padframes)
        return self


    def timestamp(self):
        """Return collected_date from json as a datetime object,          
           WARNING:  older veresion of the app do not include timezone info in this string, so this datetime is not offset aware
        """
        et = pytz.timezone("US/Eastern")                            
        return datetime.strptime(self.attributes["collected_date"], "%Y-%m-%d %H:%M:%S").astimezone(et)
    
    def uploaded(self):
        #print("[pycollector.video]: WARNING - Reporting timestamp in the JSON, which may differ from the actual time the backend processed the video")
        return self.timestamp()

    def metadata(self):
        return self._load_json().attributes

    def videoid(self):
        return self.attributes["video_id"] if "video_id" in self._load_json().attributes else None

    def collectorid(self):
        return self.attributes["collector_id"] if "collector_id" in self._load_json().attributes else None

    def subjectid(self):
        return self.attributes["subject_id"][0] if "subject_id" in self._load_json().attributes else None
        
    def collectionid(self):
        return self.attributes["collection_id"] if "collection_id" in self._load_json().attributes else None

    def collection_name(self):
        return self.attributes["collection_name"] if "collection_name" in self._load_json().attributes else None
    def collection(self):
        return self.collection_name()
    
    def duration(self):
        """Video length in seconds"""
        return float(self.attributes["duration"]) if "duration" in self._load_json().attributes else 0.0

    def quickshow(self, framerate=10, nocaption=False):
        print("[pycollector.video]: setting quickshow input framerate=%d" % framerate)
        return (
            self.fetch()
            .clone()
            .framerate(framerate)
            .mindim(256)
            .show(nocaption=nocaption)
        )
            
    def downcast(self):
        """Convert from pycollector.video to vipy.video.Scene by downcasting class"""
        v = self.clone()
        v.__class__ = Scene
        return v

    def upcast(self):
        """Convert from pycollector.video to pycollector.admin.video by upcasting class, available to admins only"""
        vipy.util.try_import('pycollector.admin.video', message="Access denied - upcast() is limited to Visym Collector admins only")
        import pycollector.admin.video
        v = self.clone()
        v.__class__ = pycollector.admin.video.Video
        return v
        
    def project(self):
        return self.attributes['project_name']

    def program(self):
        return self.attributes['program_name']

    def objectdetection(self, frame=1):
        """Run an object detector on a given frame of video.  It is more efficient to construct an ObjectDetector() object once and reuse it."""        
        from pycollector.detection import ObjectDetector
        return ObjectDetector()(self.frame(frame))

    def facedetection(self, frame=1):
        """Run face detection on a given frame of video.  It is more efficient to construct a FaceDetector() object once and reuse it."""
        from pycollector.detection import FaceDetector        
        return FaceDetector()(self.frame(frame))
    
    
def search():
    import pycollector.project
    return pycollector.project.Project(since='2020-09-01')


def last(n=1, program=None):
    import pycollector.project    

    if program:
        return pycollector.project.Project(program_id=program, since='2020-09-01', last=n).last(n)
    else:
        return pycollector.project.Project(since='2020-09-01', last=n).last(n)
