import os
import random
import vipy
from vipy.util import (
    readjson,
    isS3url,
    tempjson,
    filetail,
    tempdir,
    totempdir,
    remkdir,
    flatlist,
    tolist,
    groupbyasdict,
    writecsv,
    filebase,
    filepath,
    fileext,
    isurl,
)
from vipy.object import Track
import vipy.version

if vipy.version.is_at_least("0.8.11"):
    from vipy.activity import Activity
else:
    from vipy.object import Activity
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
)
import webbrowser
import copy
import decimal
from decimal import Decimal
import calendar
import pytz
import hashlib
from collector.review import score_verified_instance_by_id

from collector.project import Project

# Set configurble variables
co_Collectors_table = 'strCollector-hirn6lrwxfcrvl65xnxdejvftm-visym'
co_Videos_table = 'strVideos-hirn6lrwxfcrvl65xnxdejvftm-visym'
co_Collections_table = 'strCollections-hirn6lrwxfcrvl65xnxdejvftm-visym'
co_Activities_table = 'strActivities-hirn6lrwxfcrvl65xnxdejvftm-visym'
co_Instances_table = 'strInstances-hirn6lrwxfcrvl65xnxdejvftm-visym'
co_Rating_table = 'strRating-hirn6lrwxfcrvl65xnxdejvftm-visym'


# GLOBAL DDB TABLES: much faster to have them here than regenerate on every object
# However, it's not very satisfying to have this specific to a package
co_Instances_dev = collector.admin.Backend()._dynamodb_resource.Table(
    co_Instances_table
)
co_Rating = collector.admin.Backend()._dynamodb_resource.Table(co_Rating_table)


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


def applabel_to_piplabel(k=None):

    d = {"P002_P002C001_Closing": "person_closes_car_door",
         "P002_P002C001_Entering": "person_enters_car",
         "P002_P002C001_Exiting": "person_exits_car",
         "P002_P002C001_Opening": "person_opens_car_door",
         "P002_P002C002_Closing": "person_closes_car_trunk",
         "P002_P002C002_Loading": "person_loads_car",
         "P002_P002C002_Opening": "person_opens_car_trunk",
         "P002_P002C002_Unloading": "person_unloads_car",
         "P002_P002C003_Turning left": "car_turns_left",
         "P002_P002C003_Turning right": "car_turns_right",
         "P002_P002C003_U-turn": "car_makes_u_turn",
         "P002_P002C004_Picking up": "car_picks_up_person",
         "P002_P002C004_Starting": "car_starts",
         "P002_P002C004_Stopping": "car_stops",
         "P002_P002C005_Dropping off": "car_drops_off_person",
         "P002_P002C005_Starting": "car_starts",
         "P002_P002C005_Stopping": "car_stops",
         "P002_P002C006_Transferring": "person_transfers_object_to_car",
         "P002_P002C007_Reversing": "car_reverses",
         "P003_P003C001_Closing": "person_closes_motorcycle_door",  # remove me, meaningless
         "P003_P003C001_Entering": "person_enters_motorcycle",
         "P003_P003C001_Exiting": "person_exits_motorcycle",
         "P003_P003C001_Opening": "person_opens_motorcycle_door",  # remove me, meaningless
         "P003_P003C002_Closing": "person_closes_motorcycle_trunk",
         "P003_P003C002_Loading": "person_loads_motorcycle",
         "P003_P003C002_Opening": "person_opens_motorcycle_trunk",
         "P003_P003C002_Unloading": "person_unloads_motorcycle",
         "P003_P003C003_Turning left": "motorcycle_turns_left",
         "P003_P003C003_Turning right": "motorcycle_turns_right",
         "P003_P003C003_U-turn": "motorcycle_makes_u_turn",
         "P003_P003C004_Picking up": "motorcycle_picks_up_person",
         "P003_P003C004_Starting": "motorcycle_starts",
         "P003_P003C004_Stopping": "motorcycle_stops",
         "P003_P003C005_Dropping off": "motorcycle_drops_off_person",
         "P003_P003C005_Starting": "motorcycle_starts",
         "P003_P003C005_Stopping": "motorcycle_stops",
         "P003_P003C006_Transferring": "person_transfers_object_to_motorcycle",
         "P003_P003C007_Reversing": "motorcycle_reverses",
         "P004_P004C001_Closing": "person_closes_facility_door",
         "P004_P004C001_Entering": "person_enters_scene_through_structure",
         "P004_P004C001_Exiting": "person_exits_scene_through_structure",
         "P004_P004C001_Opening": "person_opens_facility_door",
         "P004_P004C002_Entering": "person_enters_scene_through_structure",
         "P004_P004C002_Exiting": "person_exits_scene_through_structure",
         "P004_P004C003_Reading": "person_reads_document",
         "P004_P004C003_Sitting": "person_sits_down",
         "P004_P004C003_Standing": "person_stands_up",
         "P004_P004C004_Dropping off": "person_puts_down_object",
         "P004_P004C004_Picking up": "person_picks_up_object",
         "P004_P004C005_Abandoning": "person_abandons_package",
         "P004_P004C005_Sets down": "person_puts_down_object",
         "P004_P004C006_Carrying": "person_carries_heavy_object",
         "P004_P004C007_Talking on": "person_talks_on_phone",
         "P004_P004C007_Texting on": "person_texts_on_phone",
         "P004_P004C008_Sitting": "person_sits_down",
         "P004_P004C008_Standing": "person_stands_up",
         "P004_P004C008_Using laptop": "person_interacts_with_laptop",
         "P004_P004C009_Purchasing": "person_purchases_from_machine",
         "P004_P004C010_Dropping off": "person_puts_down_object",
         "P004_P004C010_Picking up": "person_picks_up_object",
         "P004_P004C010_Riding": "person_rides_bicycle",
         "P005_P005C001_Talking to": "person_talks_to_person",
         "P005_P005C001_Touching hands": "hand_interacts_with_person_highfive",
         "P005_P005C002_Stealing": "person_steals_object",
         "P005_P005C003_Purchasing": "person_purchases_from_cashier",  # Remove me
         "P005_P005C004_Transferring": "person_transfers_object_to_person",
         "P005_P005C005_Hugging": "person_embraces_person",
         "P005_P005C005_Touching hands": "hand_interacts_with_person_holdhands",
         "P005_P005C006_Talking to": "person_talks_to_person",
         "P005_P005C006_Touching hands": "hand_interacts_with_person_shakehands",
    }
    return d[k] if k is not None else d


def piplabel_to_mevalabel():
    d = {v: v for (k, v) in applabel_to_piplabel().items()}

    d = {k: "hand_interacts_with_person" if "hand" in v else v for (k, v) in d.items()}
    d = {
        k: v.replace("car", "vehicle")
        if (("_car" in v or "car_" in v) and ("carries" not in v))
        else v
        for (k, v) in d.items()
    }
    d = {
        k: v.replace("motorcycle", "vehicle") if "motorcycle" in v else v
        for (k, v) in d.items()
    }
    d = {
        k: v.replace("vehicle_trunk", "trunk") if "vehicle_trunk" in v else v
        for (k, v) in d.items()
    }
    d["person_transfers_object_to_car"] = "person_transfers_object"
    d["person_transfers_object_to_motorcycle"] = "person_transfers_object"
    d["person_transfers_object_to_person"] = "person_transfers_object"
    d["person_purchases_from_machine"] = "person_purchases"
    d["person_purchases_from_cashier"] = "person_purchases"
    d = {
        k: v
        for (k, v) in d.items()
        if k
        not in set(["person_closes_motorcycle_door", "person_opens_motorcycle_door"])
    }
    return d


def mevalabel_to_index():

    return {
        "hand_interacts_with_person": 0,
        "person_abandons_package": 1,
        "person_carries_heavy_object": 2,
        "person_closes_facility_door": 3,
        "person_closes_trunk": 4,
        "person_closes_vehicle_door": 5,
        "person_embraces_person": 6,
        "person_enters_scene_through_structure": 7,
        "person_enters_vehicle": 8,
        "person_exits_scene_through_structure": 9,
        "person_exits_vehicle": 10,
        "person_interacts_with_laptop": 11,
        "person_loads_vehicle": 12,
        "person_opens_facility_door": 13,
        "person_opens_trunk": 14,
        "person_opens_vehicle_door": 15,
        "person_picks_up_object": 16,
        "person_purchases": 17,
        "person_puts_down_object": 18,
        "person_reads_document": 19,
        "person_rides_bicycle": 20,
        "person_sits_down": 21,
        "person_stands_up": 22,
        "person_steals_object": 23,
        "person_talks_on_phone": 24,
        "person_talks_to_person": 25,
        "person_texts_on_phone": 26,
        "person_transfers_object": 27,
        "person_unloads_vehicle": 28,
        "vehicle_drops_off_person": 29,
        "vehicle_makes_u_turn": 30,
        "vehicle_picks_up_person": 31,
        "vehicle_reverses": 32,
        "vehicle_starts": 33,
        "vehicle_stops": 34,
        "vehicle_turns_left": 35,
        "vehicle_turns_right": 36,
    }


def piplabel_to_index():
    return {
        "car_drops_off_person": 0,
        "car_makes_u_turn": 1,
        "car_picks_up_person": 2,
        "car_reverses": 3,
        "car_starts": 4,
        "car_stops": 5,
        "car_turns_left": 6,
        "car_turns_right": 7,
        "hand_interacts_with_person_highfive": 8,
        "hand_interacts_with_person_holdhands": 9,
        "hand_interacts_with_person_shakehands": 10,
        "motorcycle_drops_off_person": 11,
        "motorcycle_makes_u_turn": 12,
        "motorcycle_picks_up_person": 13,
        "motorcycle_reverses": 14,
        "motorcycle_starts": 15,
        "motorcycle_stops": 16,
        "motorcycle_turns_left": 17,
        "motorcycle_turns_right": 18,
        "person_abandons_package": 19,
        "person_carries_heavy_object": 20,
        "person_closes_car_door": 21,
        "person_closes_car_trunk": 22,
        "person_closes_facility_door": 23,
        "person_closes_motorcycle_trunk": 24,
        "person_embraces_person": 25,
        "person_enters_car": 26,
        "person_enters_scene_through_structure": 27,
        "person_exits_car": 28,
        "person_exits_scene_through_structure": 29,
        "person_interacts_with_laptop": 30,
        "person_loads_car": 31,
        "person_loads_motorcycle": 32,
        "person_opens_car_door": 33,
        "person_opens_car_trunk": 34,
        "person_opens_facility_door": 35,
        "person_opens_motorcycle_trunk": 36,
        "person_picks_up_object": 37,
        "person_purchases_from_machine": 38,
        "person_puts_down_object": 39,
        "person_reads_document": 40,
        "person_rides_bicycle": 41,
        "person_sits_down": 42,
        "person_stands_up": 43,
        "person_steals_object": 44,
        "person_talks_on_phone": 45,
        "person_talks_to_person": 46,
        "person_texts_on_phone": 47,
        "person_transfers_object_to_car": 48,
        "person_transfers_object_to_motorcycle": 49,
        "person_transfers_object_to_person": 50,
        "person_unloads_car": 51,
        "person_unloads_motorcycle": 52,
    }


def applabel_to_longlabel():
    """FIXME: this mapping should be baked into the app"""
    print(
        '[collector.dataset.applabel_to_longlabel]:  Scanning table "co_Activities_Mobile_ID_Dict_Dev"'
    )
    t = collector.admin.Backend()._dynamodb_resource.Table(
        "co_Activities_Mobile_ID_Dict_Dev"
    )
    return {d["Mobile_ID"]: d["Instance_ID"] for d in t.scan()["Items"]}


def get_training_videos():
    print(
        '[collector.dataset.applabel_to_longlabel]:  Scanning table "co_Activities_Mobile_ID_Dict_Dev"'
    )
    t = collector.admin.Backend()._dynamodb_resource.Table(
       co_Collections_table
    )
    return {
        d["name"]: [
            vipy.video.Video(url=url, attributes=d) for url in d["training_videos"]
        ]
        for d in t.scan()["Items"]
    }


def videoid_in_s3_not_ddb_v2():
    """ Get a set of video ids that are in S3 but not in DynamoDB

    Returns:
        [set] -- set of video ids that is in S3 but not in DynamoDB
    """

    bucketname = "diva-prod-data-lake174516-visym"
    prefix = "uploads/Programs/"

    mp4_set = set()
    json_set = set()

    for key in get_matching_s3_keys(
        s3_client=collector.admin.Backend()._s3_client,
        bucket=bucketname,
        prefix=prefix,
        suffix=(".json", ".mp4"),
    ):
        print(key)
        if ".mp4" in key:
            mp4_set.add(key.split("/")[-1].split(".")[0])
        elif ".json" in key:
            json_set.add(key.split("/")[-1].split(".")[0])

    videoid_in_s3 = mp4_set.intersection(json_set)

    P = Project(alltime=True)

    videoid_in_ddb = P.df.video_id.unique()

    return videoid_in_s3.difference(videoid_in_ddb)



# Helper function that referenced from https://alexwlchan.net/2017/07/listing-s3-keys/ Copyright © 2012–20 Alex Chan.
def get_matching_s3_keys(bucket, s3_client, prefix="", suffix=""):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param s3_client: Name of the S3 client.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """

    kwargs = {"Bucket": bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.startswith(prefix) and key.endswith(suffix):
                if obj["Size"] > 0 and obj["LastModified"].date() > yyyymmdd_to_date(
                    "2020-03-18"
                ):
                    yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


def _fix_ratings_04MAY20():
    """This function should not be run after 2020-05-11, delete me after that date"""
    P = Project(since="2020-05-04")
    items = P.ratings(reviewer="joseph.napoli@stresearch.com", badonly=True)

    print(
        "[collector.dataset._fix_ratings_04MAY20]: Flipping bad label ratings for vehicle entering and exiting that Joe N. got wrong"
    )
    print(
        "[collector.dataset._fix_ratings_04MAY20]: Scanning %d instances rated bad by Joe..."
        % len(items)
    )
    for item in items:
        if bool(item["Bad_Label"]) is True:
            try:
                i = collector.dataset.Instance(instanceid=item["id"])
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(
                    'skipping instance ID "%s" not found' % item["id"]
                )  # Ignore me, this is due to duplicates
                continue

            if (
                "P002_P002C001_Entering" in i.category()
                or "P002_P002C001_Exiting" in i.category()
            ):
                print(
                    '[collector.dataset._fix_ratings_04MAY20]: rating good for "%s"'
                    % (str(i))
                )
                i.rate(good=True)  # flip rating


