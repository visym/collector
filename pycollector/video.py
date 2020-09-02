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

import vipy
assert vipy.version.is_at_least('1.8.24')
from vipy.globals import print
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
from pycollector.globals import isapi
import pycollector.globals


class Instance(object):
    """collector.project.Instance class

       An Instance() is an observed Activity() collected as part of a CollectionInstance() and spcified by a Collection()
    """

    def __init__(self, instanceid=None, quicklookurl=None, strict=True, query=None):
        assert (quicklookurl is None or isurl(quicklookurl)), "Input must be a quicklook URL of the form: https://diva-str-prod-data-public.s3.amazonaws.com/Quicklooks/20200507_172502444708376634347184_quicklook_P004_P004C005_Abandoning_0.jpg"
        assert (quicklookurl is not None or instanceid is not None or query is not None), "Must provide one input"
        co_Instances = pycollector.globals.backend().table.instance
        
        if quicklookurl is not None:
            videoid = filebase(quicklookurl).split("_quicklook_")[0]
            instances = co_Instances.query(
                IndexName=lowerif("Video_ID-index", isapi('v2')),
                KeyConditionExpression=Key(lowerif("Video_ID", isapi('v2'))).eq(videoid),
            )["Items"]

            instances = [i for i in instances if i[lowerif("S3_Path", isapi('v2'))] == quicklookurl]
            if strict:
                assert len(instances) == 1, "Instance not found"

        elif instanceid is not None:
            instances = co_Instances.query(
                IndexName=lowerif("Instance_ID-index", isapi('v2')),
                KeyConditionExpression=Key(lowerif("Instance_ID", isapi('v2'))).eq(instanceid),
            )["Items"]
            if strict:
                assert len(instances) == 1, "Instance not found"

        elif query is not None:
            instances = [query]  # this is the output from a query to the co_Instances table

        else:
            raise ValueError("Invalid Input")

        self._instance = instances[0] if len(instances) == 1 else None
        if self.isvalid():
            self._instance = {k.lower():v for (k,v) in self._instance.items()}
        
    def __repr__(self):
        return str("<pycollector.video.instance: category=%s, videoid=%s, instanceid=%s>" % (self.shortname(), self.videoid(), self.instanceid()))

    def dict(self):
        return self._instance

    def collector(self):
        return self._instance['collector_email']

    def collection(self):
        return self._instance['collection_name']
    
    def collectionid(self):
        return self._instance['collection_id']
    
    def uploaded(self):
        return self._instance['uploaded_date']
    
    def isvalid(self):
        """There are instances that exist as quicklooks that do no exist in the instances table due to repeats, allow for non-strict loading (check with isvalid() so that we can load in bulk)"""
        return self._instance is not None

    def isperfect(self):
        return self._instance['good_for_training_score'] > 0

    def review_reason(self):
        desc = ["Wrong label" if self._instance['bad_label_score'] > 0 else "",
                "Box too big" if self._instance['bad_box_big_score'] > 0 else "",
                "Box too small" if self._instance['bad_box_small_score'] > 0 else "",
                "Box not centered" if self._instance['bad_alignment_score'] > 0 else "",                
                "Incorrect viewpoint" if self._instance['bad_viewpoint_score'] > 0 else "",
                "Incorrect timing" if self._instance['bad_timing_score'] > 0 else "",
                "Object/activity not visible" if self._instance['bad_visibility_score'] > 0 else "",
                "Repeated location" if self._instance['bad_diversity_score'] > 0 else "",
                "Unusable video" if self._instance['bad_video_score'] > 0 else ""]
        
        return [d for d in desc if len(d) > 0]

    def review_score(self):
        return self._instance['rating_score']
    
    def shortname(self):
        assert self.isvalid()
        return self._instance["activity_name"]

    def videoid(self):
        assert self.isvalid()
        return filebase(self._instance["s3_path"]).split("_quicklook_")[0]

    def instanceid(self):
        assert self.isvalid()
        return self._instance["instance_id"]

    def project(self):
        return self._instance['project_name']

    def ispractice(self):
        return 'practice' in self.project().lower()
    
    def has_rating(self):        
        return any([v>0 for (k,v) in self._instance.items() if '_score' in k])
    
    def rating(self):
        co_Rating = pycollector.globals.backend().table.rating
        video_ratings = [{k.lower():v for (k,v) in r.items()} for r in co_Rating.query(IndexName=lowerif("Video_ID-index", isapi('v2')),
                                                                                       KeyConditionExpression=Key(lowerif("Video_ID", isapi('v2'))).eq(self.videoid()))["Items"]]
        return [v for v in video_ratings if v["id"] == self.instanceid()]

    def finalized(self, state=None):
        assert isapi('v2')
        
        if state is None:
            return self._instance["rating_score_finalized"]
        else:
            assert isinstance(state, bool)
            self._instance["rating_score_finalized"] = state

            co_Instances = pycollector.globals.backend().table.instance
            co_Instances.update_item(
                Key={
                    "id": self._instance["id"],
                    "instance_id": self.instanceid(),
                },  # must provide both partition and sort key
                UpdateExpression="SET rating_score_finalized = :state",  # YUCK
                ExpressionAttributeValues={":state": state},  # DOUBLE YUCK
                ReturnValues="UPDATED_NEW",
            )
            return self

    def isgood(self, score_threshold=0.0):
        return 'rating_score' in self._instance and self._instance['rating_score'] > score_threshold

    def is_good(self, score_threshold=0.0):
        return self.isgood(score_threshold)

    def rate(self, good=None):
        assert isapi('v2')
        raise;  # FIXME
        
        assert self.isvalid()
        ratings = self.rating()
        assert len(ratings) > 0, "Rating must be present in DDB"

        for item in ratings:
            if good is not None:
                assert (
                    isinstance(good, bool) and good is True
                ), "Current functionality supports changing rating from bad to good only"
                item["up"] = Decimal(True)
                item = {
                    k: v if "bad" not in k else Decimal(False)
                    for (k, v) in item.items()
                }
                co_Rating = pycollector.globals.backend().table.rating
                co_Rating.put_item(Item=item)

                #score_verified_instance_by_id(instance_id=self.instanceid())  # FIXME

    def add_rating(self, reviewer_id, true_rating):
        assert isapi('v2')        
        co_Rating = pycollector.globals.backend().table.rating
    
        item_keys = [
            "bad_box_big",
            "bad_box_small",
            "bad_label",
            "bad_alignment",
            "bad_video",
            "bad_viewpoint",
            "bad_timing",
            "up",
        ]
        item = {}
        et = pytz.timezone("US/Eastern")
        item["week"] = lastmonday(str(datetime.now().date()))
        item["id"] = self.instanceid()
        item["video_id"] = self.videoid()
        item["updated_time"] = (datetime.now().astimezone(et).strftime("%m/%d/%Y, %I:%M:%S %p"))
        item["reviewer_id"] = reviewer_id
        for k in item_keys:
            if k in true_rating:
                item[k] = Decimal(True)
            else:
                item[k] = Decimal(False)
        co_Rating.put_item(Item=item)

        #score_verified_instance_by_id(instance_id=self.instanceid())  # FIXME

    def quicklookurl(self):
        assert pycollector.globals.isprod(), "Only valid for production environment"
        assert self.isvalid()
        return self._instance["s3_path"]

    def animated_quicklookurl(self):
        assert pycollector.globals.isprod(), "Only valid for production environment"        
        return self._instance['animation_s3_path'] if ('animation_s3_path' in self._instance and isurl(self._instance['animation_s3_path'])) else None
    
    def clip(self, padframes=0):
        """Return just the clip for this instance.  Calling quicklook() on this object should match the quicklook url."""
        assert self.isvalid()
        k_instanceindex = int(filebase(self.quicklookurl()).split("_")[-1])
        return Video(self.videoid()).activityclip(padframes=padframes)[k_instanceindex]

    def untrimmedclip(self):
        """Return just the untrimmed clip for this instance."""
        assert self.isvalid()
        k_instanceindex = int(filebase(self.quicklookurl()).split("_")[-1])
        return Video(self.videoid()).activitysplit()[k_instanceindex]

    def fullvideo(self):
        """Return the full video containing this instance"""
        assert self.isvalid()
        return Video(self.videoid())


class Video(Scene):
    """pycollector.video.Video class
    
    """

    def __init__(
        self,
        videoid=None,
        mp4file=None,
        mp4url=None,
        jsonurl=None,
        jsonfile=None,
        mindim=512,
        dt=1,
        fetch=True,
        verbose=False,
        attributes=None,
    ):
        assert (mp4file is not None or mp4url is not None or videoid is not None), "Invalid input - Must provide either mp4file or mp4url or videoid"
        assert (jsonurl is not None or jsonfile is not None or videoid is not None), "Invalid input - Must provide either jsonurl, videoid or jsonfile"
        assert mp4url is None or isS3url(mp4url), "Invalid input - mp4url must be of the form 's3://BUCKETNAME.s3.amazonaws.com/path/to/OBJECTNAME.mp4'"
        assert jsonurl is None or isS3url(jsonurl), "Invalid input - jsonurl must be of the form 's3://BUCKETNAME.s3.amazonaws.com/path/to/OBJECTNAME.json'"
        if videoid is not None:
            assert (mp4url is None and jsonurl is None), "Invalid input - must provide either videoid or URLs, not both"        
        assert vipy.version.at_least_version("0.7.4"), "vipy >= 0.7.4 required"
        
        if videoid is not None and isapi('v2'):
            v = pycollector.globals.backend().table.video.query(IndexName="video_id-index",
                                                          KeyConditionExpression=Key('video_id').eq(videoid))['Items']
            if len(v) == 0:
                raise ValueError('Video ID "%s" not found' % videoid)
            v = v[0]
            mp4url = 's3://%s.s3.amazonaws.com/%s' % (pycollector.globals.backend()._s3_bucket, v['raw_video_file_path'])
            jsonurl = 's3://%s.s3.amazonaws.com/%s' % (pycollector.globals.backend()._s3_bucket, v['annotation_file_path'])
            videoid = None
        
        elif videoid is not None and isapi('v1'):
            jsonurl = ("s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.json" % videoid) 
            mp4url = "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.mp4" % videoid
            
        elif mp4file is not None:
            videoid = mp4file.split('/')[-1].replace('.mp4','')
            
            jsonurl = ("s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.json" % videoid) 
            mp4url = "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.mp4" % videoid

        # FIXME: why is this here?
        if "VISYM_COLLECTOR_AWS_ACCESS_KEY_ID" in os.environ:
            os.environ["VIPY_AWS_ACCESS_KEY_ID"] = os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"]
        if "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY" in os.environ:
            os.environ["VIPY_AWS_SECRET_ACCESS_KEY"] = os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"]

        # Vipy constructor
        mp4url = mp4url.replace('+',' ') if mp4url is not None else mp4url  # for cut and paste from AWS console
        jsonurl = jsonurl.replace('+',' ') if jsonurl is not None else jsonurl  # for cut and paste from AWS console        
        super(Video, self).__init__(url=mp4url, filename=mp4file, attributes=attributes)

        # Video attributes
        self._quicklook_url = "https://%s.s3.amazonaws.com/Quicklooks/%%s_quicklook_%%s_%%d.jpg" % (pycollector.globals.backend().s3_bucket())  
        self._jsonurl = jsonurl
        self._jsonfile = jsonfile
        self._dt = dt
        self._is_json_loaded = None
        self._mindim = mindim
        self._verbose = verbose
        if fetch:
            self._load_json()
            
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

            if isapi('v1'):
                d["metadata"]["collected_date"] = uploaded.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                et = pytz.timezone("US/Eastern")                
                d["metadata"]["collected_date"] = uploaded.astimezone(et).strftime("%Y-%m-%d %H:%M:%S")
                

        else:
            print('[pycollector.video]: empty JSON "%s" - SKIPPING' % jsonfile)
            d = None

        # Valid collection?
        if not pycollector.globals.backend().collection().isvalid(d["metadata"]["collection_id"]):
            print('[collector.video]: invalid collection "%s" - SKIPPING' % d["metadata"]["collection_id"])
            d = None

        # Import JSON into scene
        if d is not None:
            collection_name = pycollector.globals.backend().collection().id_to_name(d["metadata"]["collection_id"])
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
                    print('[pycollector.Video]: Removing %d bad keyboxes "%s" for videoid=%s' % (len(badboxes), str(badboxes), d["metadata"]["video_id"]))
                if len(badboxes) == len(keyboxes):
                    raise ValueError("all keyboxes in track are invalid")

                t = Track(
                    category=obj["label"],
                    framerate=float(d["metadata"]["frame_rate"]),
                    keyframes=[f for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()],
                    boxes=[bb for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()],
                    boundary="strict",
                )

                if vipy.version.is_at_least("0.8.3"):
                    self.add(t, rangecheck=False)  # no rangecheck since all tracks are guarnanteed to be within image rectangle
                else:
                    self.add(t)
                d_trackid_to_track[t.id()] = t

            # Import activities
            for a in d["activity"]:
                try:
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
                    shortlabel = shortlabel.lower()

                    category = pycollector.globals.backend().collection()[d["metadata"]["collection_id"]].shortname_to_activity(a["label"])
                    self.add(vipy.activity.Activity(category=category,
                                                    shortlabel=shortlabel,
                                                    startframe=int(a["start_frame"]),
                                                    endframe=int(a["end_frame"]),
                                                    tracks=d_trackid_to_track,
                                                    framerate=d["metadata"]["frame_rate"],
                                                    attributes=d["metadata"]))
                    
                    if not vipy.version.is_at_least("0.8.3"):
                        if int(a["start_frame"]) >= int(a["end_frame"]):
                            raise ValueError(
                                "degenerate start/end frames %d/%d",
                                int(a["start_frame"]),
                                int(a["end_frame"]),
                            )

                except Exception as e:
                    print(
                        '[pycollector.Video]: Filtering invalid activity "%s" with error "%s" for videoid=%s'
                        % (str(a), str(e), d["metadata"]["video_id"])
                    )

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
            self._is_json_loaded = False

        # Resample tracks
        if self._dt > 1:
            self.trackmap(lambda t: t.resample(self._dt))
        
        return self

    def geolocation(self):
        assert 'ipAddress' in self.metadata()
        url = 'http://api.geoiplookup.net/?query=%s' % self.metadata()['ipAddress']
        with urllib.request.urlopen(url) as f:
            response = f.read().decode('utf-8')
        d = xmltodict.parse(response)
        return dict(d['ip']['results']['result'])  
        
    def fetch(self, ignoreErrors=False):
        """Download JSON and MP4 if not already downloaded"""
        try:
            self.fetchjson()
            if vipy.version.is_at_least("0.8.8"):
                super(Video, self).fetch()
        except Exception as e:
            if ignoreErrors:
                print(
                    '[pycollector.video]: Download failed with error "%s" - SKIPPING'
                    % (str(e))
                )
            else:
                raise e
        except KeyboardInterrupt:
            raise
        return self

    def fetchvideo(self, ignoreErrors=False):
        super(Video, self).fetch()
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

    def __repr__(self):
        return str("<pycollector.video: uploaded=%s, activities=%s, scene=%s>" % (self.timestamp().strftime("%Y-%m-%d %H:%M")
                                                                                  if self.timestamp() is not None
                                                                                  else str(None),
                                                                                  self.activity_categories(),
                                                                                  str(super(Video, self).__repr__())))

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

    def sanitize(self):
        """Replace identifying email addresses 'me@here.com' with a truncated sha1 hash.  This is repeatable, so the same email address gets the same hash (not including email typos!)"""
        email = self.metadata()["collector_id"]
        if is_email_address(email):
            hash = hashlib.sha1(email.encode("UTF-8")).hexdigest()
            self.attributes["collector_id"] = hash[0:10]
        return self

    def obfuscate(self):
        """Replace identifying email addresses 'me@here.com' with 'm****@here.com.  For stronger privacy, use self.sanitize()"""
        email = self.metadata()["collector_id"]
        if is_email_address(email):
            email = "%s****@%s" % (email[0], email.split("@")[-1])
            self.attributes["collector_id"] = email
        return self

    def stabilize(self, mindim=256):
        assert vipy.version.is_at_least('0.8.13')
        from vipy.flow import Flow
        return Flow().stabilize(self.clone().mindim(mindim))

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
        if isapi('v1'):
            return (
                datetime.strptime(self.attributes["collected_date"], "%Y-%m-%d %H:%M:%S")
                if "collected_date" in self._load_json().attributes
                else None
            )
        else:
            et = pytz.timezone("US/Eastern")                            
            return datetime.strptime(self.attributes["collected_date"], "%Y-%m-%d %H:%M:%S").astimezone(et)

    def uploaded(self):
        print("[pycollector.video]: WARNING - Reporting timestamp in the JSON, which may differ from the actual time the backend processed the video")
        return self.timestamp()

    def metadata(self):
        return self._load_json().attributes

    def videoid(self):
        return (
            self.attributes["video_id"]
            if "video_id" in self._load_json().attributes
            else None
        )

    def collectorid(self):
        return (
            self.attributes["collector_id"]
            if "collector_id" in self._load_json().attributes
            else None
        )

    def collector(self):
        return self.collectorid()

    def collectionid(self):
        return (
            self.attributes["collection_id"]
            if "collection_id" in self._load_json().attributes
            else None
        )

    def collection(self):
        return self.collectionid()

    def duration(self):
        """Video length in seconds"""
        return (
            float(self.attributes["duration"])
            if "duration" in self._load_json().attributes
            else 0.0
        )

    def save_annotation(self, outdir=None):
        outfile = (
            totempdir("%s.mp4" % self.videoid())
            if outdir is None
            else os.path.join(outdir, "%s.mp4" % self.videoid())
        )
        return self.annotate().saveas(outfile).filename()        

    def quicklookurls(self, show=False):
        assert pycollector.globals.isprod(), "Quicklook URLs only available in production environment"
        
        urls = [
            self._quicklook_url % (self.videoid(), a.category(), k)
            for (k, a) in enumerate(self._load_json().activities().values())
        ]
        if show:
            assert vipy.version.is_at_least("0.8.2")
            vipy.visualize.urls([url for url in urls], display=True)
        return urls

    def quickshow(self, framerate=10, nocaption=False):
        print("[pycollector.video]: setting quickshow input framerate=%d" % framerate)
        return (
            self.fetch()
            .clone()
            .framerate(framerate)
            .mindim(256)
            .show(nocaption=nocaption)
        )

    def rating(self):
        """Return all instance ratings for this video"""

        # FIXME: the association of rating to instance depends on the quicklook ordering, which depends on a python dict, which is not
        # guararanteed to be order preserving.  If you are running python 3.7 this does not matter.
        assert vipy.version.is_at_least("0.8.0")

        co_Rating = pycollector.globals.backend().table.rating
        video_ratings = co_Rating.query(
            IndexName=lowerif("Video_ID-index", isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", isapi('v2'))).eq(self.videoid()),
        )["Items"]

        co_Instances = pycollector.globals.backend().table.instance
        instance_ratings = []
        for v in video_ratings:
            rating = co_Instances.query(
                IndexName=lowerif("Instance_ID-index", isapi('v2')),
                ProjectionExpression=lowerif("Bad_Label_Score, Bad_Viewpoint_Score, Instance_ID, Review_Reason, Verified, Rating_Score, Bad_Timing_Score, S3_Path, Bad_Box_Big_Score,  Bad_Box_Small_Score", isapi('v2')),
                KeyConditionExpression=Key(lowerif("Instance_ID", isapi('v2'))).eq(v[lowerif("ID", isapi('v2'))]),
            )["Items"][0]
            rating.update(v)
            instance_ratings.append(rating)
        return instance_ratings

    def isgood(self):
        """'Good' is defined as a video with at least one instance rated good in the video"""
        co_Rating = pycollector.globals.backend().table.rating        
        ratings = co_Rating.query(
            IndexName=lowerif("Video_ID-index", isapi('v2')), 
            ProjectionExpression=lowerif("Up", isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", isapi('v2'))).eq(self.videoid()),
        )["Items"]
        return any([lowerif("Up", isapi('v2')) in r and r[lowerif("Up", isapi('v2'))] > 0 for r in ratings])
            
    def downcast(self):
        """Convert from collector.video to vipy.video.Scene by downcasting class"""
        v = self.clone().sanitize()
        v.__class__ = Scene
        return v

    def instances(self):
        """Return all instances for this video"""
        co_Instances = pycollector.globals.backend().table.instance
        instances = co_Instances.query(
            IndexName=lowerif("Video_ID-index", isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", isapi('v2'))).eq(self.videoid()),
        )["Items"]            
        return [Instance(query=i, strict=False) for i in instances]

    def instance(self, k):
        return self.instances()[k]




