import os
import random
import vipy
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key, Attr
import copy
import decimal
from decimal import Decimal
import calendar
import pytz
import hashlib
import uuid
import urllib

import vipy
assert vipy.version.is_at_least('1.8.24')
from vipy.globals import print
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

import pycollector.util
from pycollector.util import (
    allmondays_since,
    yyyymmdd_to_date,
    is_email_address,
    isday,
    is_more_recent_than,
    nextday,
    lastmonday,
    lowerif,
    timestamp,
    fromdate, ismonday, fromtimestamp
)
import pycollector.globals
from pycollector.globals import isapi
from pycollector.video import Video, Instance
from pycollector.backend import CollectionInstance


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
        before=None,
        week=None
    ):

        co_Program = pycollector.globals.backend().table.program
        co_Video = pycollector.globals.backend().table.video        
        
        self._projects = None
        self._programid = program_id
        if program_id != "MEVA":
            response = co_Program.query(KeyConditionExpression=Key(lowerif("ID", isapi('v2')).eq(program_id)))
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
                '[pycollector.project]: Scan the entire database and return every video uploaded for program "%s" since 2020-07-20...'
                % program_id
            )

            fe = Attr(lowerif("Uploaded_Date", isapi('v2'))).gte("2020-03-18")  # Remove junk data

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
            assert isday(since), (
                "Invalid date input - use 'YYYY-MM-DD' not '%s'" % since
            )
            assert is_more_recent_than(since, "2020-03-18")
            assert before is None or (
                isday(before) and is_more_recent_than(before, since)
            ), "Invalid before date"
            print(
                "[pycollector.project]: Scan database and return every video uploaded %s"
                % (
                    "from %s to %s (inclusive)" % (since, before)
                    if before is not None
                    else "since %s" % (since)
                )
            )
            fe = Attr(lowerif("Uploaded_Date", isapi('v2'))).gte(since)  # Remove junk data
            if before is not None and isday(before):
                fe = fe & Attr(lowerif("Uploaded_Date", isapi('v2'))).lte(
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
            assert is_more_recent_than(since, "2020-03-18")
            print(
                "[pycollector.project]:  Return every video uploaded in the last %d months, %d weeks, %d days (since %s)"
                % (
                    monthsago if monthsago is not None else 0,
                    weeksago if weeksago is not None else 0,
                    daysago if daysago is not None else 0,
                    since,
                )
            )
            fe = Attr(lowerif("Uploaded_Date", isapi('v2'))).gte(since)
            
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

                fe = Attr(lowerif("Video_ID", isapi('v2'))).is_in(video_ids)
                response = co_Video.scan(FilterExpression=fe)
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
        
        print("[pycollector.project]:  Returned %d videos" % len(self.df))

    def __repr__(self):
        return str("<pycollector.project: program=%s, videos=%d, since=%s, collectors=%d>" % (self._programid,
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
        assert isapi('v1'), "Migrate me to v2"
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
        assert isapi('v1'), "Migrate me to v2"
        
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
            set([x for a in (self.df.activities_list if isapi('v2') else self.df.activities) for x in tolist(a)])
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
        assert ismonday(week)
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if row["week"] == week])
        return self
    
    def _filter_collector(self, collector_id):
        #assert is_email_address(collector_id)
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if row["collector_id"] == collector_id])
        return self

    def collector(self, collector_id):
        P = self.filter(collector=collector_id)
        print(
            '[pycollector.project]:  Filter by collector "%s" returned %d videos'
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

        assert isday(mindate), "Date must be 'YYYY-MM-DD' string"
        assert yyyymmdd_to_date(mindate) >= yyyymmdd_to_date(self._since), (
            "Date '%s' must be greater than or equal to the date range of constructor '%s'"
            % (mindate, self._since)
        )
        self.df = pd.DataFrame([row for (index, row) in self.df.iterrows() if pd.to_datetime(row["uploaded_date"]).astimezone(et).date() >= yyyymmdd_to_date(mindate)])
        
        if maxdate is not None:
            assert isday(maxdate), "Date must be 'YYYY-MM-DD' string"
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
        co_Instances = pycollector.globals.backend().table.instance
        instances_result = []
        for sub_week in self.df['week'].unique():
            response = co_Instances.query(
                IndexName=lowerif("week-uploaded_date-index", isapi('v2')),
                KeyConditionExpression=Key(lowerif("Week", isapi('v2'))).eq(sub_week),
            )
            instances_result.extend(response["Items"])  # may be empty      
            while ("LastEvaluatedKey" in response and response["LastEvaluatedKey"] is not None):
                response = co_Instances.query(
                    IndexName=lowerif("week-uploaded_date-index", isapi('v2')),
                    KeyConditionExpression=Key(lowerif("Week", isapi('v2'))).eq(sub_week),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                instances_result.extend(response["Items"])  # may be empty                

        return [ Instance(query=instance, strict=False) for instance in instances_result if instance['video_id'] in self.videoid()]


    def ratings(self, reviewer=None, badonly=False):
        """Query ratings table for videoids in this project, and filter by reviewer.  If badonly=True, then return only reviews that were bad for some reason"""

        co_Rating = pycollector.globals.backend().table.rating
        
        # Iterate over ratings table, query each video ID in the current project, with pagination
        ratinglist = []
        for sub_week in self.df['week'].unique():
            response = co_Rating.query(
                IndexName=lowerif("week-index", isapi('v2')),  # "week" for ratings table is derived from instance upload not when rating was given
                KeyConditionExpression=Key(lowerif("Week", isapi('v2'))).eq(sub_week),
            )
            ratinglist.extend(response["Items"])  # may be empty            

            while ("LastEvaluatedKey" in response and response["LastEvaluatedKey"] is not None):
                response = co_Rating.query(
                    IndexName=lowerif("week-index", isapi('v2')),
                    KeyConditionExpression=Key(lowerif("Week", isapi('v2'))).eq(sub_week),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                ratinglist.extend(response["Items"])  # may be empty                
                
        ratings = sorted(
            [{k: v for (k, v) in r.items()} for r in ratinglist],
            key=lambda x: x[lowerif("Week", isapi('v2'))],
        )
        if reviewer is not None:
            assert is_email_address(reviewer)
            ratings = [r for r in ratings if r[lowerif("Reviewer_ID", isapi('v2'))] == reviewer]
        if badonly is True:
            ratings = [r for r in ratings if r[lowerif("Up", isapi('v2'))] == False]

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


    def quickhtml(self, outfile=None, display=True):
        """Generate a quicklook HTML file to show all quicklooks in the current project.  This is useful for filtering by collector and regenerating quicklooks on demand.  
           Note that this regenrates the quicklooks on demand and does not pull from the backend.
        """
        assert vipy.version.is_at_least("0.8.2")
        if len(self) > 100:
            warnings.warn(
                "[pycollector.project]: Generating quickhtml for %d videos, This may take a while ..."
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


def search():
    """Return all videos for a python user"""

    # this should:
    # ask the user to login, triggered by accessing pycollector.backend.Backend() without credentials
    # If the user has previously logged in, we save a credentials.pkl file containing their access token so that this does not need login the next time
    # query the video table for this user using the pycollector.video.Project() tools
    # Return a list of pycollector.video.Video() objects
    
    pass
    
