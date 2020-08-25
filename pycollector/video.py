import os
import random
import vipy
from vipy.util import (
    readjson,
    isS3url,
    tempjson,
    tempdir,
    totempdir,
    remkdir,
    flatlist,
    tolist,
    groupbyasdict,
    writecsv,
    filebase,
    filetail,
    filepath,
    fileext,
    isurl,
    tolist
)
from vipy.object import Track
import vipy.version

from vipy.activity import Activity as vipy_Activity
from vipy.video import Scene
from vipy.geometry import BoundingBox
import vipy.downloader
import vipy.version
import warnings
from datetime import datetime, timedelta
import numpy as np
import collector.admin
import pandas as pd
from boto3.dynamodb.conditions import Key, Attr
import collector.util
from collector.util import (
    allmondays_since,
    yyyymmdd_to_date,
    is_email_address,
    isday,
    is_more_recent_than,
    nextday,
    lastmonday,
    lowerif,
    timestamp,
    fromdate, ismonday
)
from collector.ddb_util import delete_by_data
import copy
import decimal
from decimal import Decimal
import calendar
import pytz
import hashlib
from collector.review import score_verified_instance_by_id
import uuid
from collector.gsheets import Gsheets
import collector 
from collector.workforce import Collectors
import urllib



class Instance(object):
    """collector.project.Instance class

       An Instance() is an observed Activity() collected as part of a CollectionInstance() and spcified by a Collection()
    """

    def __init__(self, quicklookurl=None, instanceid=None, strict=True, query=None):
        assert (quicklookurl is None or
                isurl(quicklookurl)), "Input must be a quicklook URL of the form: https://diva-str-prod-data-public.s3.amazonaws.com/Quicklooks/20200507_172502444708376634347184_quicklook_P004_P004C005_Abandoning_0.jpg"
        assert (quicklookurl is not None or instanceid is not None or query is not None), "Must provide one input"

        if quicklookurl is not None:
            videoid = filebase(quicklookurl).split("_quicklook_")[0]
            instances = co_Instances.query(
                IndexName=lowerif("Video_ID-index", collector.version.isapi('v2')),
                KeyConditionExpression=Key(lowerif("Video_ID", collector.version.isapi('v2'))).eq(videoid),
            )["Items"]

            instances = [i for i in instances if i[lowerif("S3_Path", collector.version.isapi('v2'))] == quicklookurl]
            if strict:
                assert len(instances) == 1, "Instance not found"

        elif instanceid is not None:
            instances = co_Instances.query(
                IndexName=lowerif("Instance_ID-index", collector.version.isapi('v2')),
                KeyConditionExpression=Key(lowerif("Instance_ID", collector.version.isapi('v2'))).eq(instanceid),
            )["Items"]
            if strict:
                assert len(instances) == 1, "Instance not found"

        elif query is not None:
            instances = [
                query
            ]  # this is the output from a query to the co_Instances table

        else:
            raise ValueError("Invalid Input")

        self._instance = instances[0] if len(instances) == 1 else None
        if self.isvalid():
            self._instance = {k.lower():v for (k,v) in self._instance.items()}
        
    def __repr__(self):
        return str(
            "<collector.project.Instance: category=%s, videoid=%s, instanceid=%s>"
            % (self.shortname(), self.videoid(), self.instanceid())
        )

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
        video_ratings = [{k.lower():v for (k,v) in r.items()} for r in co_Rating.query(IndexName=lowerif("Video_ID-index", collector.version.isapi('v2')),
                                                                                       KeyConditionExpression=Key(lowerif("Video_ID", collector.version.isapi('v2'))).eq(self.videoid()))["Items"]]
        return [v for v in video_ratings if v["id"] == self.instanceid()]

    def finalized(self, state=None):
        assert collector.version.api() == 'v2'
        
        if state is None:
            return self._instance["rating_score_finalized"]
        else:
            assert isinstance(state, bool)
            self._instance["rating_score_finalized"] = state

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
        return self._instance['rating_score'] > score_threshold

    def is_good(self, score_threshold=0.0):
        return self.isgood(score_threshold)

    def rate(self, good=None):
        assert collector.version.api() == 'v2'        

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
                co_Rating.put_item(Item=item)
                score_verified_instance_by_id(instance_id=self.instanceid())

    def add_rating(self, reviewer_id, true_rating):
        assert collector.version.api() == 'v2'                

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
        item["updated_time"] = (
            datetime.now().astimezone(et).strftime("%m/%d/%Y, %I:%M:%S %p")
        )
        item["reviewer_id"] = reviewer_id
        for k in item_keys:
            if k in true_rating:
                item[k] = Decimal(True)
            else:
                item[k] = Decimal(False)
        co_Rating.put_item(Item=item)
        score_verified_instance_by_id(instance_id=self.instanceid())

    def quicklookurl(self):
        assert self.isvalid()
        return self._instance["s3_path"]

    def animated_quicklookurl(self):
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
    """collector.project.Video class
    
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
        assert (
            mp4file is not None or mp4url is not None or videoid is not None
        ), "Invalid input - Must provide either mp4file or mp4url or videoid"
        assert (
            jsonurl is not None or jsonfile is not None or videoid is not None
        ), "Invalid input - Must provide either jsonurl, videoid or jsonfile"
        assert mp4url is None or isS3url(
            mp4url
        ), "Invalid input - mp4url must be of the form 's3://BUCKETNAME.s3.amazonaws.com/path/to/OBJECTNAME.mp4'"
        assert jsonurl is None or isS3url(
            jsonurl
        ), "Invalid input - jsonurl must be of the form 's3://BUCKETNAME.s3.amazonaws.com/path/to/OBJECTNAME.json'"
        if videoid is not None:
            assert (mp4url is None and jsonurl is None), "Invalid input - must provide either videoid or URLs, not both"        
        assert vipy.version.at_least_version("0.7.4"), "vipy >= 0.7.4 required"
        
        if videoid is not None and 'v2' in collector.version.api():
            v = co_Video.query(IndexName="video_id-index",
                               KeyConditionExpression=Key('video_id').eq(videoid))['Items']
            if len(v) == 0:
                raise ValueError('Video ID "%s" not found' % videoid)
            v = v[0]
            mp4url = 's3://%s.s3.amazonaws.com/%s' % (s3_bucket, v['raw_video_file_path'])
            jsonurl = 's3://%s.s3.amazonaws.com/%s' % (s3_bucket, v['annotation_file_path'])
            videoid = None
        
        elif videoid is not None and collector.version.isapi('v1'):
            jsonurl = (
                "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.json" % videoid
            ) 
            mp4url = "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.mp4" % videoid
        elif mp4file is not None:
            videoid = mp4file.split('/')[-1].replace('.mp4','')
            
            jsonurl = (
                "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.json" % videoid
            ) 
            mp4url = "s3://diva-prod-data-lake.s3.amazonaws.com/temp/%s.mp4" % videoid


            
        if "VISYM_COLLECTOR_AWS_ACCESS_KEY_ID" in os.environ:
            os.environ["VIPY_AWS_ACCESS_KEY_ID"] = os.environ[
                "VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"
            ]
        if "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY" in os.environ:
            os.environ["VIPY_AWS_SECRET_ACCESS_KEY"] = os.environ[
                "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"
            ]

        # Constructor
        mp4url = mp4url.replace('+',' ') if mp4url is not None else mpp4url  # for cut and paste from AWS console
        jsonurl = jsonurl.replace('+',' ') if jsonurl is not None else jsonurl  # for cut and paste from AWS console        
        super(Video, self).__init__(url=mp4url, filename=mp4file, attributes=attributes)

        # Video attributes
        self._quicklook_url = "https://diva-str-prod-data-public.s3.amazonaws.com/Quicklooks/%s_quicklook_%s_%d.jpg"
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
                print('[collector.Video]:  Parsing "%s"' % jsonfile)

            d = readjson(jsonfile)
            if "collection_id" not in d["metadata"]:
                d["metadata"]["collection_id"] = d["metadata"][
                    "video_id"
                ]  # android 1.1.1(3) bug

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
                d["metadata"][
                    "rotate"
                ] = "rot90ccw"  # iOS (7) bug, different than iOS (6)

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

            if collector.version.isapi('v1'):
                d["metadata"]["collected_date"] = uploaded.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                et = pytz.timezone("US/Eastern")                
                d["metadata"]["collected_date"] = uploaded.astimezone(et).strftime("%Y-%m-%d %H:%M:%S")
                

        else:
            print('[collector.project.Video]: empty JSON "%s" - SKIPPING' % jsonfile)
            d = None

        # Import JSON into scene
        if d is not None:
            self.category("%s" % (d["metadata"]["collection_id"]))
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
                    print(
                        '[collector.Video]: Removing %d bad keyboxes "%s" for videoid=%s'
                        % (len(badboxes), str(badboxes), d["metadata"]["video_id"])
                    )
                if len(badboxes) == len(keyboxes):
                    raise ValueError("all keyboxes in track are invalid")

                t = Track(
                    category=obj["label"],
                    framerate=float(d["metadata"]["frame_rate"]),
                    keyframes=[
                        f for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()
                    ],
                    boxes=[bb for (f, bb) in zip(keyframes, keyboxes) if bb.isvalid()],
                    boundary="strict",
                )

                if vipy.version.is_at_least("0.8.3"):
                    self.add(
                        t, rangecheck=False
                    )  # no rangecheck since all tracks are guarnanteed to be within image rectangle
                else:
                    self.add(t)
                d_trackid_to_track[t.id()] = t

            # Import activities
            for a in d["activity"]:
                try:
                    if (
                        d["metadata"]["collection_id"] == "P004C009"
                        and d["metadata"]["device_identifier"] == "android"
                    ):
                        shortlabel = "Buying (Machine)"
                    elif (
                        d["metadata"]["collection_id"] == "P004C008"
                        and d["metadata"]["device_identifier"] == "ios"
                        and "Purchasing" in a["label"]
                    ):
                        # BUG: iOS (11) reports wrong collection id for "purchase something from a machine" as P004C008 instead of P004C009
                        shortlabel = "Buying (Machine)"
                    elif (
                        d["metadata"]["collection_id"] == "P004C009"
                        and d["metadata"]["device_identifier"] == "ios"
                    ):
                        # BUG: iOS (11) reports wrong collection id for "pickup and dropoff with bike messenger" as P004C009 instead of P004C010
                        shortlabel = a["label"]  # unchanged
                    elif d["metadata"]["collection_id"] == "P005C003":
                        shortlabel = "Buying (Cashier)"
                    else:
                        shortlabel = a["label"]
                    shortlabel = shortlabel.lower()

                    self.add(
                        vipy_Activity(
                            category="%s_%s_%s"
                            % (
                                d["metadata"]["project_id"],
                                d["metadata"]["collection_id"],
                                a["label"],
                            ),  # will be remapped when needed
                            shortlabel=shortlabel,
                            startframe=int(a["start_frame"]),
                            endframe=int(a["end_frame"]),
                            tracks=d_trackid_to_track,
                            framerate=d["metadata"]["frame_rate"],
                            attributes=d["metadata"],
                        )
                    )
                    if not vipy.version.is_at_least("0.8.3"):
                        if int(a["start_frame"]) >= int(a["end_frame"]):
                            raise ValueError(
                                "degenerate start/end frames %d/%d",
                                int(a["start_frame"]),
                                int(a["end_frame"]),
                            )

                except Exception as e:
                    print(
                        '[collector.Video]: Filtering invalid activity "%s" with error "%s" for videoid=%s'
                        % (str(a), str(e), d["metadata"]["video_id"])
                    )

            if d["metadata"]["rotate"] == "rot90ccw":
                self.rot90ccw()
            elif d["metadata"]["rotate"] == "rot90cw":
                self.rot90cw()

            self._is_json_loaded = True

            # Minimum dimension of video for reasonably fast interactions (must happen after JSON load to get frame size from JSON)
            if self._mindim is not None:
                if "frame_width" in self.metadata():  # older JSON bug
                    self.rescale(
                        self._mindim
                        / min(
                            int(self.metadata()["frame_width"]),
                            int(self.metadata()["frame_height"]),
                        )
                    )  # does not require load
                else:
                    assert vipy.version.is_at_least("0.8.0")
                    self.clear()  # remove this old video from consideration
        else:
            self._is_json_loaded = False

        # Resample tracks
        if self._dt > 1:
            self.trackmap(lambda t: t.resample(self._dt))
        
        return self

    def geolocation(self):
        assert 'ipAddress' in self.metadata()
        import xmltodict  
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
                    '[collector.video.fetch]: Download failed with error "%s" - SKIPPING'
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
                print('[collector.Video]:  Fetching "%s"' % self._jsonurl)
                try:
                    vipy.downloader.s3(self._jsonurl, self._jsonfile)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(
                        '[collector.project.Video]: S3 download error "%s" - SKIPPING'
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
        return str("<collector.Video: collector=%s, uploaded=%s, activities=%s, scene=%s>" % (str(self._load_json().collectorid()),
                                                                                              self.timestamp().strftime("%Y-%m-%d %H:%M")
                                                                                              if self.timestamp() is not None
                                                                                              else str(None),
                                                                                              self.activity_categories(),
                                                                                              str(super(Video, self).__repr__())))

    def activity_categories(self):
        """Return a set of unique activity categories in the video, not including object categories"""
        self._load_json()
        return set([a.category() for a in self._load_json().activities().values()])

    def get_instance_geometrics_info(self):
        """
        return 
        
        """
        return [
            a.boundingbox().xywh()
            + (a.boundingbox().xywh()[2] * a.boundingbox().xywh()[3],)
            for clip in self._load_json().activityclip()
            for a in clip.activities().values()
        ]

    def quicklooks(self, n=9, dilate=1.5, mindim=256, fontsize=10, context=True):
        """Return a vipy.image.Image object containing a montage quicklook for each of the activities in this video.  
        
        Usage:
         
        >>> filenames = [im.saveas('/path/to/quicklook.jpg') for im in self.quicklooks()]
        
        """
        assert vipy.version.is_at_least("0.8.2")
        print('[collector.project.quicklooks]: Generating quicklooks for video "%s"' % self.videoid())
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
        if collector.version.isapi('v1'):
            return (
                datetime.strptime(self.attributes["collected_date"], "%Y-%m-%d %H:%M:%S")
                if "collected_date" in self._load_json().attributes
                else None
            )
        else:
            et = pytz.timezone("US/Eastern")                            
            return datetime.strptime(self.attributes["collected_date"], "%Y-%m-%d %H:%M:%S").astimezone(et)

    def uploaded(self):
        print("[collector.project.Video]: WARNING - Reporting timestamp in the JSON, which may differ from the actual time the backend processed the video")
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
        urls = [
            self._quicklook_url % (self.videoid(), a.category(), k)
            for (k, a) in enumerate(self._load_json().activities().values())
        ]
        if show:
            assert vipy.version.is_at_least("0.8.2")
            vipy.visualize.urls([url for url in urls], display=True)
        return urls

    def quickshow(self, framerate=10, nocaption=False):
        print("[collector.project.Video]: setting quickshow input framerate=%d" % framerate)
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

        video_ratings = co_Rating.query(
            IndexName=lowerif("Video_ID-index", collector.version.isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", collector.version.isapi('v2'))).eq(self.videoid()),
        )["Items"]

        instance_ratings = []
        for v in video_ratings:
            rating = co_Instances.query(
                IndexName=lowerif("Instance_ID-index", collector.version.isapi('v2')),
                ProjectionExpression=lowerif("Bad_Label_Score, Bad_Viewpoint_Score, Instance_ID, Review_Reason, Verified, Rating_Score, Bad_Timing_Score, S3_Path, Bad_Box_Big_Score,  Bad_Box_Small_Score", collector.version.isapi('v2')),
                KeyConditionExpression=Key(lowerif("Instance_ID", collector.version.isapi('v2'))).eq(v[lowerif("ID", collector.version.isapi('v2'))]),
            )["Items"][0]
            rating.update(v)
            instance_ratings.append(rating)
        return instance_ratings

    def isgood(self):
        """'Good' is defined as a video with at least one instance rated good in the video"""
        ratings = co_Rating.query(
            IndexName=lowerif("Video_ID-index", collector.version.isapi('v2')), 
            ProjectionExpression=lowerif("Up", collector.version.isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", collector.version.isapi('v2'))).eq(self.videoid()),
        )["Items"]
        return any([lowerif("Up", collector.version.isapi('v2')) in r and r[lowerif("Up", collector.version.isapi('v2'))] > 0 for r in ratings])
            
    def downcast(self):
        """Convert from collector.project.Video to vipy.video.Scene by downcasting class"""
        v = self.clone().sanitize()
        v.__class__ = Scene
        return v

    def instances(self):
        """Return all instances for this video"""
        instances = co_Instances.query(
            IndexName=lowerif("Video_ID-index", collector.version.isapi('v2')), 
            KeyConditionExpression=Key(lowerif("Video_ID", collector.version.isapi('v2'))).eq(self.videoid()),
        )["Items"]            
        return [Instance(query=i, strict=False) for i in instances]

    def instance(self, k):
        return self.instances()[k]


        
class Project(object):
    """collector.project.Project class

       Projects() are sets of CollectionInstances() and Instances() in a program.
    """
    
    def __init__(
        self,
        program_id="MEVA",
        project_id=None,
        weeksago=None,
        monthsago=None,
        daysago=5,
        since=None,
        alltime=False,
        Video_IDs=None,
        backend=collector.admin.Backend(),
        before=None,
        week=None
    ):

        self._backend = backend
        self._projects = None
        self._programid = program_id
        if program_id != "MEVA":
            response = co_Program.query(KeyConditionExpression=Key(lowerif("ID", collector.version.isapi('v2')).eq(program_id)))
            if response["Count"] == 0:
                raise ValueError('Unknown programid "%s"' % program_id)
            assert response["Count"] == 1
            raise ValueError('FIXME')

        if week is not None:
            assert ismonday(week)
            since = week
            before = (fromdate(since) + timedelta(days=6)).strftime("%Y-%m-%d")
            
        if alltime:
            print(
                '[collector.project.Project]: Scan the entire database and return every video uploaded for program "%s" since 2020-07-20...'
                % program_id
            )
            self._backend = backend

            fe = Attr(lowerif("Uploaded_Date", collector.version.isapi('v2'))).gte("2020-03-18")  # Remove junk data

            response = co_Video.scan(FilterExpression=fe)
            items = response["Items"]
            while (
                "LastEvaluatedKey" in response
                and response["LastEvaluatedKey"] is not None
            ):
                response = co_Video.scan(
                    FilterExpression=fe, ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response["Items"])

            self.df = pd.DataFrame(items)
            self._since = "2020-03-18"

        elif since is not None:
            assert (
                monthsago is None and weeksago is None
            ), "Invalid input - must specify only since"
            assert collector.util.isday(since), (
                "Invalid date input - use 'YYYY-MM-DD' not '%s'" % since
            )
            assert collector.util.is_more_recent_than(since, "2020-03-18")
            assert before is None or (
                isday(before) and is_more_recent_than(before, since)
            ), "Invalid before date"
            print(
                "[collector.project.Project]: Scan database and return every video uploaded %s"
                % (
                    "from %s to %s (inclusive)" % (since, before)
                    if before is not None
                    else "since %s" % (since)
                )
            )
            fe = Attr(lowerif("Uploaded_Date", collector.version.isapi('v2'))).gte(since)  # Remove junk data
            if before is not None and isday(before):
                fe = fe & Attr(lowerif("Uploaded_Date", collector.version.isapi('v2'))).lte(
                    nextday(before)
                )  # inclusive endpoint, add one day

            response = co_Video.scan(FilterExpression=fe)
            items = response["Items"]
            while (
                "LastEvaluatedKey" in response
                and response["LastEvaluatedKey"] is not None
            ):
                response = co_Video.scan(
                    FilterExpression=fe, ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response["Items"])

            self.df = pd.DataFrame(items)
            self._since = since

        elif monthsago is not None or weeksago is not None or daysago is not None:
            days = daysago if daysago is not None else 0
            days += 7 * weeksago if weeksago is not None else 0
            days += (365 / 12.0) * monthsago if monthsago is not None else 0
            since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            assert collector.util.is_more_recent_than(since, "2020-03-18")
            print(
                "[collector.project.Project]:  Return every video uploaded in the last %d months, %d weeks, %d days (since %s)"
                % (
                    monthsago if monthsago is not None else 0,
                    weeksago if weeksago is not None else 0,
                    daysago if daysago is not None else 0,
                    since,
                )
            )
            fe = Attr(lowerif("Uploaded_Date", collector.version.isapi('v2'))).gte(since)
            
            response = co_Video.scan(FilterExpression=fe)
            items = response["Items"]
            while (
                "LastEvaluatedKey" in response
                and response["LastEvaluatedKey"] is not None
            ):
                response = co_Video.scan(
                    FilterExpression=fe, ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response["Items"])

            self.df = pd.DataFrame(items)
            self._since = since

        elif Video_IDs:

            # TODO - paginating
            n = 99
            video_ids_list = [Video_IDs[i : i + n] for i in range(0, len(Video_IDs), n)]

            dataframes = []

            for video_ids in video_ids_list:

                fe = Attr(lowerif("Video_ID", collector.version.isapi('v2'))).is_in(video_ids)
                response = vo_Video.scan(FilterExpression=fe)
                items = response["Items"]
                while (
                    "LastEvaluatedKey" in response
                    and response["LastEvaluatedKey"] is not None
                ):
                    response = co_Video.scan(
                        FilterExpression=fe,
                        ExclusiveStartKey=response["LastEvaluatedKey"],
                    )
                    items.extend(response["Items"])

                dataframes.append(pd.DataFrame(items))

            self.df = pd.concat((dataframes))
            self._since = since

        else:
            raise ValueError("Invalid date range selected")

        # Canonicalize dataframe to lowercase
        self.df = self.df.rename(columns={k:k.lower() for k in self.df.columns})
        self.df = self.df.loc[:,~self.df.columns.duplicated()]  # FIXME: dedupe 'bad_box_big_score' in DDB
        
        print("[collector.project.Project]:  Returned %d videos" % len(self.df))

    def __repr__(self):
        return str("<collector.project.Project: program=%s, videos=%d, since=%s, collectors=%d>" % (self._programid,
                                                                                                    len(self.videoid()),
                                                                                                    str(self._since),
                                                                                                    len(self.collectorid())))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, k):
        if isinstance(k, int):
            return Video(self.df.iloc[k].video_id)
        elif isinstance(k, slice):
            return [Video(j) for j in self.df.iloc[k].video_id]
        else:
            raise ValueError('Invalid slice "%s"' % (str(slice)))
        
    def _s3_bucketpath_to_url(self, path):
        """Convert BUCKETNAME/path/to/objectname.ext -> s3://BUCKETNAME.s3.amazonaws.com/path/to/objectname.ext"""
        assert collector.version.isapi('v1'), "Migrate me to v2"
        (bucketname, objectname) = path.split("/", 1)
        return "s3://%s.s3.amazonaws.com/%s" % (bucketname, objectname)

    def since(self):
        return self._since
    
    def clone(self):
        c = copy.copy(self)
        c.df = self.df.copy(deep=True)
        return c
    
    def take(self, n):
        return [self[k] for k in random.sample(range(0, len(self)), n)]

    def collectioninstances(self):
        return [CollectionInstance(collection_name=r.collection_name,
                                   video_id=r.video_id,
                                   collector=r.collector_email,
                                   table=r.to_dict())
                for (k, r) in self.df.iterrows()]
                
    def projects(self):
        assert collector.version.isapi('v1'), "Migrate me to v2"
        
        if self._projects is not None:
            return self._projects

        t = self._backend._dynamodb_resource.Table("co_Projects")
        response = t.query(KeyConditionExpression=Key("ID").eq(self._programid))
        projectlist = set([r["Name"] for r in response["Items"]])
        d = {}
        for p in projectlist:
            d[p] = {}
            t = self._backend._dynamodb_resource.Table("co_Collections")
            response = t.query(
                KeyConditionExpression=Key("ID").eq("%s_%s" % (self._programid, p))
            )
            collections = [r["Name"] for r in response["Items"]]
            for c in collections:
                t = self._backend._dynamodb_resource.Table("co_Activities")
                response = t.query(
                    KeyConditionExpression=Key("ID").eq(
                        "%s_%s_%s" % (self._programid, p, c)
                    )
                )
                activities = [r["Name"] for r in response["Items"]]
                d[p][c] = activities

        self._projects = d
        return d

    def collectorID(self):
        return set(self.df.collector_id) if len(self.df) > 0 else []

    def collectorid(self):
        return self.collectorID()

    def collectoremail(self):
        return set(self.df.collector_email) if len(self.df) > 0 else []        
    
    def collectors(self):
        return self.collectorID()

    def collectionID(self):
        return set(self.df.collection_id)

    def collectionid(self):
        return self.collectionID()

    def collections(self):
        return self.collectionID()

    def collectioncount(self):
        return {
            k: len(v)
            for (k, v) in groupbyasdict(self.df.collection_id, lambda x: x).items()
        }

    def uploaded(self):
        et = pytz.timezone("US/Eastern")
        return (
            [pd.to_datetime(x).astimezone(et) for x in set(self.df.uploaded_date)]
            if len(self.df) > 0
            else []
        )

    def activities(self):
        return (
            set([x for a in (self.df.activities_list if collector.version.isapi('v2') else self.df.activities) for x in tolist(a)])
            if len(self.df) > 0
            else set()
        )

    def videoID(self):
        return set(self.df.video_id) if len(self) > 0 else set([])

    def videoIDduplicates(self):
        return [
            k
            for (k, v) in groupbyasdict(self.videoid(), lambda x: x).items()
            if len(v) > 1
        ]

    def videoid(self):
        return self.videoID()

    def topandas(self):
        return self.df

    def tolist(self):
        """Return a list of dictionaries"""
        return [row.to_dict() for (index, row) in self.df.iterrows()]

    def filter(self, collector=None, since=None, collection=None, verified=None, before=None, week=None):
        """Return a Project() clone that has been filtered according to the requested fields, dates are inclusive of endpoints, [since, before]"""
        f = self.clone()
        f._filter_week(week=lastmonday(week)) if (week is not None) else f        
        f._filter_date(mindate=since, maxdate=before) if (since is not None or before is not None) else f
        f._filter_collector(collector) if collector is not None else f
        f._filter_collection(collection) if collection is not None else f
        f._filter_verified(verified) if verified is not None else f
        return f

    def _filter_week(self, week):
        assert collector.util.ismonday(week)
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if row["week"] == week])
        return self
    
    def _filter_collector(self, collector_id):
        #assert collector.util.is_email_address(collector_id)
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if row["collector_id"] == collector_id])
        return self

    def collector(self, collector_id):
        P = self.filter(collector=collector_id)
        print(
            '[collector.project.Project]:  Filter by collector "%s" returned %d videos'
            % (collector_id, len(P))
        )
        return P

    def _filter_collection(self, collection_id):
        """Return only the provided collection id"""
        assert collection_id is not None
        self.df = pd.DataFrame(
            [
                row
                for (index, row) in self.df.iterrows()
                if collection_id in row["collection_id"]
            ]
        )
        return self

    def _filter_not_collections(self, collectionid_list):
        """Remove all elements from project in the collectionid_list"""
        self.df = pd.DataFrame([row
                                for (index, row) in self.df.iterrows()
                                if row["collection_id"] not in set(collectionid_list)])
        return self

    def _filter_not_collection_name(self, collection_name):
        """Remove all elements from project not equal to the given collection"""
        self.df = pd.DataFrame([row
                                for (index, row) in self.df.iterrows()
                                if row["collection_name"] == collection_name])
        return self
    
    def _filter_collections(self, collectionid_list):
        """Return all elements from project in the collectionid_list"""
        self.df = pd.DataFrame([row
                                for (index, row) in self.df.iterrows()
                                if row["collection_id"] in set(collectionid_list)])
        return self

    def _filter_videoid(self, videoid_list):
        """Remove all vidoes in the videoid list"""
        self.df = pd.DataFrame([row
                                for (index, row) in self.df.iterrows()
                                if row["video_id"] not in set(tolist(videoid_list))])
        return self

    def _filter_not_videoid(self, videoid_list):
        """Remove all vidoes not in the videoid list"""
        self.df = pd.DataFrame([row
                                for (index, row) in self.df.iterrows()
                                if row["video_id"] in set(tolist(videoid_list))])
        return self
    
        
    def _filter_date(self, mindate, maxdate=None):
        """mindate should be of the form YYYY-MM-DD such as 2020-03-30 for March 30, 2020"""
        et = pytz.timezone("US/Eastern")
        mindate = mindate if mindate is not None else self._since
        mindate = mindate if mindate != "today" else str(datetime.now().astimezone(et).date())
        mindate = mindate if mindate != "yesterday" else str((datetime.now().astimezone(et) - timedelta(days=1)).date())
        mindate = mindate if mindate != "thisweek" else str((datetime.now().astimezone(et) - timedelta(days=7)).date())

        assert collector.util.isday(mindate), "Date must be 'YYYY-MM-DD' string"
        assert yyyymmdd_to_date(mindate) >= yyyymmdd_to_date(self._since), (
            "Date '%s' must be greater than or equal to the date range of constructor '%s'"
            % (mindate, self._since)
        )
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if pd.to_datetime(row["uploaded_date"]).astimezone(et).date() >= yyyymmdd_to_date(mindate)])
        
        if maxdate is not None:
            assert collector.util.isday(maxdate), "Date must be 'YYYY-MM-DD' string"
            assert yyyymmdd_to_date(maxdate) >= yyyymmdd_to_date(self._since), (
                "Date '%s' must be greater than or equal to the date range of constructor '%s'"
                % (maxdate, self._since)
            )
            self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if pd.to_datetime(row["uploaded_date"]).astimezone(et).date() <= yyyymmdd_to_date(maxdate)])

        self._since = mindate
        return self

    def _filter_activity(self, category):
        self.df = pd.DataFrame(
            [
                row
                for (index, row) in self.df.iterrows()
                if category in row["activities"]
            ]
        )
        return self

    def _filter_verified(self, verified=True):
        assert isinstance(verified, bool) or verified is None
        if verified is True:
            self.df = pd.DataFrame(
                [row for (index, row) in self.df.iterrows() if row["rating_score"] > 0]
            )
        elif verified is False:
            self.df = pd.DataFrame(
                [row for (index, row) in self.df.iterrows() if row["rating_score"] == 0]
            )
        return self

    def _filter_unverified(self):
        return self._filter_verified(verified=False)

    def _filter_videoid(self, videoid):
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if row["video_id"] not in videoid])
        return self
        
    def mean_duration(self):
        """Mean video length of project in seconds, requires parsing JSON"""
        return np.mean([v.duration() for v in self.videos()])

    def dedupe(self, vidlist):
        seen = set()
        outlist = []
        for v in vidlist:
            if v.hasactivities() and v.videoid() not in seen:
                seen.add(v.videoid())
                outlist.append(v)
        return outlist

    def videos(self, collector=None, since=None, collection=None, verified=None, fetch=True):
        """Cache JSON files and create a list of collector.project.Videos"""
        f = self.filter(collector, since, collection, verified)
        return [
            Video(vid, fetch=fetch)
            for (vid, date) in sorted(
                zip(f.df.video_id, f.df.uploaded_date),
                key=lambda x: pd.to_datetime(x[1]),
                reverse=True,
            )
        ]  # lazy load

    def videoscores(self):
        return {d['video_id']:{k:v for (k,v) in d.items() if '_score' in k} for d in self.tolist()}
        
    def verified_videos(self, collector=None, since=None, collection=None, fetch=True):
        """Cache JSON files and create a list of collector.project.Videos that have passed verification"""
        # FIXME: this should really go into filter
        return self.videos(collector, since, collection, verified=True, fetch=fetch)

        
    def instances(self):
        instances_result = []
        for sub_week in self.df['week'].unique():
            response = co_Instances.query(
                IndexName=lowerif("week-uploaded_date-index", collector.version.isapi('v2')),
                KeyConditionExpression=Key(lowerif("Week", collector.version.isapi('v2'))).eq(sub_week),
            )
            instances_result.extend(response["Items"])  # may be empty      
            while ("LastEvaluatedKey" in response and response["LastEvaluatedKey"] is not None):
                response = co_Instances.query(
                    IndexName=lowerif("week-uploaded_date-index", collector.version.isapi('v2')),
                    KeyConditionExpression=Key(lowerif("Week", collector.version.isapi('v2'))).eq(sub_week),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                instances_result.extend(response["Items"])  # may be empty                

        return [ Instance(query=instance, strict=False) for instance in instances_result if instance['video_id'] in self.videoid()]


    def ratings(self, reviewer=None, badonly=False):
        """Query ratings table for videoids in this project, and filter by reviewer.  If badonly=True, then return only reviews that were bad for some reason"""

        # Iterate over ratings table, query each video ID in the current project, with pagination
        ratinglist = []
        for sub_week in self.df['week'].unique():
            response = co_Rating.query(
                IndexName=lowerif("week-index", collector.version.isapi('v2')),  # "week" for ratings table is derived from instance upload not when rating was given
                KeyConditionExpression=Key(lowerif("Week", collector.version.isapi('v2'))).eq(sub_week),
            )
            ratinglist.extend(response["Items"])  # may be empty            

            while ("LastEvaluatedKey" in response and response["LastEvaluatedKey"] is not None):
                response = co_Rating.query(
                    IndexName=lowerif("week-index", collector.version.isapi('v2')),
                    KeyConditionExpression=Key(lowerif("Week", collector.version.isapi('v2'))).eq(sub_week),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                ratinglist.extend(response["Items"])  # may be empty                
                
        ratings = sorted(
            [{k: v for (k, v) in r.items()} for r in ratinglist],
            key=lambda x: x[lowerif("Week", collector.version.isapi('v2'))],
        )
        if reviewer is not None:
            assert collector.util.is_email_address(reviewer)
            ratings = [r for r in ratings if r[lowerif("Reviewer_ID", collector.version.isapi('v2'))] == reviewer]
        if badonly is True:
            ratings = [r for r in ratings if r[lowerif("Up", collector.version.isapi('v2'))] == False]

        return [{k.lower():v for (k,v) in r.items()} for r in ratings]  # canonicalize

    def instanceratings(self, reviewer=None, badonly=False):
        """Return instances for all ratings for this project, filtered by reviewer and badonly"""
        return [Instance(instanceid=r["id"], strict=False) for r in self.ratings(reviewer, badonly)]

    
    def bestcollectors(self):
        """Return number of submitted videos per collector, sorted in decreasing order"""
        return sorted(
            [
                (k, len(v))
                for (k, v) in groupbyasdict(
                    [
                        row
                        for (index, row) in self.df.iterrows()
                        if row["collector_id"] in self.collectors()
                    ],
                    lambda x: x["collector_id"],
                ).items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def collectorfeedback(self, outfile=None, header=True):
        """Return the number of uploaded videos and ratings score for the time period in the constructor"""
        csv = sorted(
            [
                [k, len(v), np.mean([float(r["rating_score"]) for r in v])]
                for (k, v) in groupbyasdict(
                    [
                        row
                        for (index, row) in self.df.iterrows()
                        if row["collector_id"] in self.collectors()
                    ],
                    lambda x: x["collector_id"],
                ).items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        csv = (
            [
                [
                    "# collectorid",
                    "num_uploaded (since %s)" % self._since,
                    "mean rating score (since %s)" % self._since,
                ]
            ]
            + csv
            if header
            else csv
        )
        return csv if outfile is None else writecsv(csv, outfile)

    def stats(self, refresh=False):
        """Return a dictionary of useful statistics about this project for dashboarding.  Requires accessing the Collections()"""

        # Totals: by week
        data = {'byweek':[], 'alltime':None}
        for w in allmondays_since(self.since()):
            P = self.filter(week=w)
            if len(P) > 0:
                cinst = P.collectioninstances()
                inst = P.instances()
                d_byweek = {'Week':w,
                            'Active Collectors': len(P.collectors()),
                            'Collected Videos': len(cinst),
                            'Rated Videos': sum([v.has_rating() for v in cinst]),
                            'Verified Videos': sum([v.is_good() for v in cinst]),
                            'Collected Instances': len(inst),
                            'Rated Instances': sum([i.has_rating() for i in inst]),
                            'Verified Instances': sum([i.is_good() for i in inst]),                      
                            'Last Updated':timestamp()}                      
                data['byweek'].append(d_byweek)

        # Totals: alltime        
        cinst = self.collectioninstances()
        inst = self.instances()
        data['alltime'] = {'Active Collectors': len(self.collectors()),
                           'Collected Videos': len(cinst),
                           'Rated Videos': sum([v.has_rating() for v in cinst]),
                           'Verified Videos': sum([v.is_good() for v in cinst]),
                           'Collected Instances': len(inst),
                           'Rated Instances': sum([i.has_rating() for i in inst]),
                           'Verified Instances': sum([i.is_good() for i in inst]),                      
                           'Last Updated':timestamp()}

        # Totals: by category (requires activity to collection query for this campaign)
        inst = self.instances()
        C = collector.project.Collections()
        d_collected_bycategory = vipy.util.countby(inst, lambda i: C[i.collection()].shortname_to_activity(i.shortname()))
        d_rated_bycategory = vipy.util.countby([i for i in inst if i.has_rating()], lambda i: C[i.collection()].shortname_to_activity(i.shortname()))
        d_verified_bycategory = vipy.util.countby([i for i in inst if i.is_good()], lambda i: C[i.collection()].shortname_to_activity(i.shortname()))
        data['bycategory'] = {'collected':d_collected_bycategory,
                              'rated':d_rated_bycategory,
                              'verified':d_verified_bycategory}

        # Ratings refresh (sanity check the rating score)
        return data
    
        
    def collectordashboard(
        self,
        outfile=None,
        header=True,
        Project_IDs=[],
        Collection_IDs=[],
        ex_Project_IDs=[],
        ex_Collection_IDs=[],
    ):
        """Return a CSV of (collector_id, uploaded_date, rating_score, rating_reason activity_name, quicklook URL) for the time range defined in the constructor. Used to sync the collector dashboard
           This function does not use any of the data in Project(), so it does not really need to be in here.         
        """
 
        # This function is for legacy dashboarding only.       
        warnings.warn('collector.project.Project().collectordashboard() has been deprecated')
        
        C = collector.admin.Instance()
        assert (
            len(allmondays_since(self._since)) <= 5
        ), "Too many instances requested, keep this to one month"
        instances = [
            C.get_activity_instances(week=w) for w in allmondays_since(self._since)
        ]  # iterated query by date

        R = collector.admin.Report()

        activity_long_names_dict = R.get_activity_long_names_dict()

        def _f_badrating_to_description(
            badlabel,
            box,
            badbox_big,
            badbox_small,
            badviewpoint,
            badtiming,
            badalignment,
            badvisibility,
            baddiversity,
            badvideo,
        ):
            desc = [
                "label" if badlabel > 0 else "",
                "box" if box > 0 else "",
                "box(too big)" if badbox_big > 0 else "",
                "box(too small)" if badbox_small > 0 else "",
                "viewpoint" if badviewpoint > 0 else "",
                "timing" if badtiming > 0 else "",
                "subject centric alignment" if badalignment > 0 else "",
                "visibility for objects" if badvisibility > 0 else "",
                "diversity for settings and activities" if baddiversity > 0 else "",
                "video content" if badvideo > 0 else "",
            ]
            desc = [d for d in desc if len(d) > 0]

            desc = (
                "incorrect " + " and ".join(desc)
                if len(desc) > 0
                else "verification error"
            )

            return desc.rstrip()

        et = pytz.timezone("US/Eastern")
        (mindate, maxdate) = (
            min(self.uploaded()),
            max(self.uploaded()),
        )  # time range from project (ET)
        csv = [
            [
                r["collecgtor_id" if collector.version.isapi('v1') else "collector_email"],
                pd.to_datetime(r["uploaded_date"]).date().isoformat(),  # FIXME: EST?
                "good"
                if float(r["rating_score"]) > 0
                else "needs improvement"
                if r["verified"]
                else "Pending for verification"
                if r["project_id"] not in ex_Project_IDs
                else "Video not from the valid Projects"
                if r["collection_id"] not in ex_Collection_IDs
                else "Video not from the valid Collection",
                "good"
                if float(r["rating_score"]) > 0
                else _f_badrating_to_description(
                    float(r["bad_label_score"]),
                    float(r["bad_box_score"]),
                    float(r["bad_box_big_score"]),
                    float(r["bad_box_small_score"]),
                    float(r["bad_viewpoint_score"]),
                    float(r["bad_timing_score"]),
                    float(r["bad_alignment_score"]),
                    float(r["bad_visibility_score"]),
                    float(r["bad_diversity_score"]),
                    float(r["bad_video_score"]),
                )
                if r["verified"]
                else "Pending for verification"
                if r["project_id"] not in ex_Project_IDs
                else "Video not from the valid Projects"
                if r["collection_id"] not in ex_Collection_IDs
                else "Video not from the valid Collection",
                activity_long_names_dict[r["id"]],
                '=HYPERLINK("%s")' % (r["s3_path"]),
            ]
            for i in instances
            for (index, r) in i.iterrows()
            if pd.to_datetime(r["uploaded_date"]).astimezone(et) >= mindate
            and pd.to_datetime(r["uploaded_date"]).astimezone(et) <= maxdate
        ]
        csv = sorted(csv, key=lambda x: x[0])

        csv = (
            [
                [
                    "collectorid",
                    "uploaded",
                    "rating",
                    "rating reason",
                    "activity name",
                    "quicklook URL",
                ]
            ]
            + csv
            if header
            else csv
        )
        return csv if outfile is None else writecsv(csv, outfile)

    def quickhtml(self, outfile=None, display=True):
        """Generate a quicklook HTML file to show all quicklooks in the current project.  This is useful for filtering by collector and regenerating quicklooks on demand.  
           Note that this regenrates the quicklooks on demand and does not pull from the backend.
        """
        assert vipy.version.is_at_least("0.8.2")
        if len(self) > 100:
            warnings.warn(
                "[collector.project.quickhtml]: Generating quickhtml for %d videos, This may take a while ..."
                % len(self)
            )
        vipy.visualize.tohtml(
            [
                im.setattributes(v.metadata())
                for v in self.videos()
                for im in v.fetch().quicklooks()
            ],
            title="QuickHTML",
            display=display,
            outfile=outfile,
            attributes=True,
        )

    def quicklookurls(self, outfile=None, display=True):
        """Generate a standalong HTML file containing the quicklook URLs for the current filtered project"""
        assert vipy.version.is_at_least("0.8.2")
        urls = [url for v in self.videos() for url in v.fetchjson().quicklookurls()]
        vipy.visualize.urls(urls, outfile=outfile, display=display)
        return urls

    def analysis(self, outdir=None):
        """Return a set of useful statistics and plots for this dataset, saved in outdir"""
        import vipy.metrics

        outdir = tempdir() if outdir is None else outdir
        videos = self.videos()  # fetch JSON
        ratings = [r for r in self.ratings()]
        activities = [a for v in videos for a in v.activitylist()]
        tracks = [t for v in videos for t in v.tracklist()]
        uploaded = self.uploaded()
        d_newcategory = applabel_to_longlabel()
        activities = [a.category(d_newcategory[a.category()]) for a in activities]

        d = {
            "since": self._since,
            "program": "MEVA",
            "num_videos": len(self.videoid()),
            "mean_duration": self.mean_duration(),
            "bestcollectors": self.bestcollectors()[0:3],
            "ratings": {
                "good": len([r for r in ratings if "up" in r and r["up"] is True]),
                "good_for_training": len([r for r in ratings if "good_for_training" in r and r["good_for_training"] is True]),
                "bad_box_big": len(
                    [
                        r
                        for r in ratings
                        if "bad_box_big" in r and r["bad_box_big"] is True
                    ]
                ),
                "bad_box_small": len(
                    [
                        r
                        for r in ratings
                        if "bad_box_small" in r and r["bad_box_small"] is True
                    ]
                ),
                "bad_label": len(
                    [r for r in ratings if "bad_label" in r and r["bad_label"] is True]
                ),
                "bad_alignment": len(
                    [
                        r
                        for r in ratings
                        if "bad_alignment" in r and r["bad_alignment"] is True
                    ]
                ),
                "bad_video": len(
                    [r for r in ratings if "bad_video" in r and r["bad_video"] is True]
                ),
                "bad_viewpoint": len(
                    [
                        r
                        for r in ratings
                        if "bad_viewpoint" in r and r["bad_viewpoint"] is True
                    ]
                ),
                "bad_timing": len(
                    [
                        r
                        for r in ratings
                        if "bad_timing" in r and r["bad_timing"] is True
                    ]
                ),
                "bad_visibility": len(
                    [
                        r
                        for r in ratings
                        if "bad_visibility" in r and r["bad_visibility"] is True
                    ]
                ),
                "bad_diversity": len(
                    [
                        r
                        for r in ratings
                        if "bad_diversity" in r and r["bad_diversity"] is True
                    ]
                ),
            },
        }

        d["activity_categories"] = set([a.category() for a in activities])
        d["object_categories"] = set([t.category() for t in tracks])
        d["num_activities"] = sorted(
            [
                (k, len(v))
                for (k, v) in groupbyasdict(activities, lambda a: a.category()).items()
            ],
            key=lambda x: x[1],
        )

        # Histogram of instances
        (categories, freq) = zip(*reversed(d["num_activities"]))
        barcolors = ["blue" if not "vehicle" in c else "green" for c in categories]
        d["num_activities_histogram"] = vipy.metrics.histogram(
            freq,
            categories,
            barcolors=barcolors,
            outfile=os.path.join(outdir, "num_activities_histogram.pdf"),
            ylabel="Collected Instances",
        )

        # Pie chart of ratings
        labels = [k for (k, v) in d["ratings"].items()]
        counts = [d["ratings"][k] for k in labels]
        explode = [0.1 if k == "good" else 0.0 for k in labels]
        d["ratings_pie"] = vipy.metrics.pie(
            counts,
            labels,
            explode=explode,
            outfile=os.path.join(outdir, "ratings_pie.pdf"),
        )

        # Histogram of upload times for videos
        (categories, freq) = zip(
            *[
                ("%d:00" % k, len(v))
                for (k, v) in groupbyasdict(uploaded, lambda t: t.hour).items()
            ]
        )
        d["uploaded_time_histogram"] = vipy.metrics.histogram(
            freq,
            categories,
            barcolors=None,
            outfile=os.path.join(outdir, "uploaded_time_histogram.pdf"),
            ylabel="Collected Videos",
            title="Hour of day (EDT)",
        )

        # Histogram of upload weekdays for videos
        (categories, freq) = zip(
            *[
                (calendar.day_name[k], len(v))
                for (k, v) in groupbyasdict(
                    uploaded, lambda t: t.date().weekday()
                ).items()
            ]
        )
        d["uploaded_weekday_histogram"] = vipy.metrics.histogram(
            freq,
            categories,
            barcolors=None,
            outfile=os.path.join(outdir, "uploaded_weekday_histogram.pdf"),
            ylabel="Collected Videos",
            xrot=45,
        )

        return d
