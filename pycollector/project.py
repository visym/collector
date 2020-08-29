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

#import collector.admin
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
    fromdate, ismonday
)
#from collector.ddb_util import delete_by_data
#from collector.review import score_verified_instance_by_id

#from collector.gsheets import Gsheets
#import collector 
#from collector.workforce import Collectors
import pycollector.globals
from pycollector.globals import isapi
from pycollector.video import Video, Instance


class CollectorAssignment(object):
    """ class for collector assignment

    Args:
        object ([type]): [description]
    """

    def __init__(self, program_name=None, project_name=None,):
        
        self.program_name = program_name
        self.project_name = project_name

        # Get Program information
        self.program = Program(name=self.program_name)
        # Get Project information
        self.project = ProgramProjects()

        # Set program and project dataframe
        program_df = self.program.to_df()
        project_df = self.project.get_all_projects_in_df()

        self._program_df = program_df[['name','program_id']]
        self._project_df = project_df[['project_id','name']]

        self._program_df.rename(columns={"name" : "program_name"}, inplace=True)
        self._project_df.rename(columns={"name": "project_name"}, inplace=True)

        # Get Collections
        self._collections = Collections()
        collections_df = self._collections.to_df()

        # Filter for active collections
        self._collections_df = collections_df[collections_df.active == True]

        # Get all active collectors
        dashboard = collector.dashboard.MevaDashboard()
        self._active_collector_emails = dashboard.active_collectors()
        
        # Get all collectors from DDB
        collectors = Collectors()
        all_collectors_df = collectors.get_collectors(as_dataframe=True)
        active_collectors_df = all_collectors_df[all_collectors_df.collector_email.isin(self._active_collector_emails)]

        # Filtered by is_consented
        active_collectors_df = active_collectors_df[active_collectors_df['is_consented'] == True]
        # Fill NaN with 0
        active_collectors_df.fillna(0, inplace=True)

        # Get active_collectors_ids
        self._active_collectors_ids = list(active_collectors_df.collector_id)

        # Set co_Collections_Collectors_df
        co_Collections_Collectors_df = pd.merge(self._collections_df, self._program_df, how='left', on='program_name')
        co_Collections_Collectors_df = pd.merge(self._collections_df, self._project_df, how='left', on='project_name')

        # Filtered by program and project 
        if program_name:
            co_Collections_Collectors_df = co_Collections_Collectors_df[co_Collections_Collectors_df['program_name'] == program_name]
        if project_name:
            co_Collections_Collectors_df = co_Collections_Collectors_df[co_Collections_Collectors_df['project_name'] == project_name]

        # TEMP for now - TODO remove these once we recreate the new collections with new attributes
        co_Collections_Collectors_df['isTrainingVideoEnabled'] = co_Collections_Collectors_df['show_training_video'] 
        co_Collections_Collectors_df['isConsentRequired'] = True
        co_Collections_Collectors_df['consent_overlay_text'] = 'Please select the record button, say "I consent to this video collection”'

        co_Collections_Collectors_df['assigned_date'] = datetime.now(pytz.timezone("US/Eastern")).isoformat()
        co_Collections_Collectors_df.rename(columns={'name':'collection_name'}, inplace=True)
        co_Collections_Collectors_df.drop(columns=['id'],inplace=True)

        # Set assignment 
        collection_assignment_dfs =[] 

        for c_id in self._active_collectors_ids:
            this_co_Collections_df_filtered = co_Collections_Collectors_df.copy()
            this_co_Collections_df_filtered['collector_id'] = c_id
            collection_assignment_dfs.append(this_co_Collections_df_filtered)
        
        self.collection_assignment_df = pd.concat(collection_assignment_dfs)
        self.collection_assignment_df.fillna('None',inplace=True)


    def delete_current_assignments_in_DDB(self):
        """ Batch delete current collectioin assiignment to DDB 
        """
        delete_by_data(co_Collections_assignment_table,self.collection_assignment_df.to_dict(orient='records'),PKey='collector_id', SKey='collection_name')


    def batch_insert_collection_assignment_to_DDB(self):
        """ Batch insert current collectioin assiignment to DDB 
        """
        # Batch Update with insert
        with co_Collections_assignment_table.batch_writer() as batch:
            for row in self.collection_assignment_df.iterrows():
                item = row[1].to_dict()
                batch.put_item(Item=item)


    
class ProgramProject(object):
    """ collector.project.ProgramProject

        A single project definition, as specified in the Projects table.
    """
    
    def __init__(self, item=None, name=None):
        self._item = [k for k in co_Projects.scan()["Items"] if k['name'] == name][0] if name is not None else item
        assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
        
    def __repr__(self):
        return str('<collector.project.ProgramProject: "%s", name=%s, project_id=%s,  created_date=%s>' % (self._item['name'], self._item['name'], self._item['program_id'], self._item['created_date']))

    def dict(self):
        return self._item

    def to_df(self):
        return pd.DataFrame([self._item])


class ProgramProjects(object):
    """ collector.project.ProgramProjects

        An interface to the Projects table
    """
        
    def __init__(self):
        self._itemdict = {k['name']:k for k in co_Projects.scan()["Items"]}
    
    def dict(self):
        return self._itemdict

    def __getitem__(self, name):
        assert name in self._itemdict, "Unknown project name '%s'" % name
        return ProgramProject(self._itemdict[name])

    def get_all_projects_in_df(self):
        return pd.DataFrame([ v for k, v in  self._itemdict.items()])

    def new(self, program_name, project_name):
        """ Add new project (if not present). Check with the latest state in DynamoDB

        Args:
            name (str): name of the program
            client (str): the client associate with the program
        """
        # Add new project (if not present)
        # Check with the latest state of programs in DDB 
        response = co_Projects.query(KeyConditionExpression=Key("id").eq(program_name))
        if not any([x['name'] == project_name for x in response['Items']]):
            item = {'id':program_name,
                    'name':project_name,
                    'created_date':timestamp(),
                    'mobile_id':project_name,
                    'project_id':str(uuid.uuid4())}
            co_Projects.put_item(Item=item)    
            print("Created New Project: ", project_name)
        else:
            print("This project %s is already exists. " % name)    

            

class Program(object):
    """ collector.project.Programs

        An interface to the Programs table
    """

    class _Program(object):
        """A single program definition"""
    
        def __init__(self, itemdict):
            self._item = itemdict  #[k for k in co_Program.scan()["Items"] if k['name'] == name][0] if name is not None else item
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
        
        def __repr__(self):
            return str('<pycollector.project.Program: "%s", name=%s, program_id=%s, client=%s, created_date=%s>' % (self._item['name'], self._item['name'], self._item['program_id'], self._item['client'], self._item['created_date']))

        def dict(self):
            return self._item

        def to_df(self):
            return pd.DataFrame([self._item])

    
    def __init__(self, tabledict):
        self._itemdict = {k['name']:k for k in tabledict}
    
    def dict(self):
        return self._itemdict

    def __getitem__(self, name):
        assert name in self._itemdict, "Unknown program name '%s'" % name
        return self._Program(self._itemdict[name])

    def new(self, name, client):
        """ Add new program (if not present). Check with the latest state in DynamoDB

        Args:
            name (str): name of the program
            client (str): the client associate with the program
        """
        # Add new program (if not present)
        # Check with the latest state of programs in DDB
        co_Program = pycollector.globals.backend().table.program
        
        response = co_Program.query(KeyConditionExpression=Key("id").eq(name))
        if not any([x['name'] == name for x in response['Items']]):
            item = {'id':name,
                    'name':name,
                    'client':client,
                    'created_date':timestamp(),
                    'program_id':str(uuid.uuid4())}
            co_Program.put_item(Item=item)  
        else:
            print("This program %s is already exists. " % name)          

        
class Activity(object):
    """ collector.project.Activity class

        An interface to the Activity table which defines the relationship between collections and activities.
    """


    class _Activity(object):
        """A single activity definition, as specified in the Activities table.
        """
    
        def __init__(self, itemdict):
            self._item = itemdict
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
        
        def __repr__(self):
            return str('<pycollector.project.Activity: "%s", shortname=%s, id=%s, project=%s, collection=%s>' % (self._item['name'], self._item['short_name'], self._item['activity_id'], self._item['project_name'], self._item['collection_name']))

        def dict(self):
            return self._item

        def enable(self):
            self._item['active'] = True
            self._item['updated_date'] = timestamp()        
            pycollector.globals.backend().table.activity.put_item(Item=self._item)
            return self

        def disable(self):
            self._item['active'] = False
            self._item['updated_date'] = timestamp()        
            pycollector.globals.backend().table.activity.put_item(Item=self._item)
            return self

        def short_name(self, name=None):
            if name is not None:
                assert isinstance(name, str), "shortn name must be a string"
                self._item['short_name'] = name
                self._item['updated_date'] = timestamp()        
                pycollector.globals.backend().table.activity.put_item(Item=self._item)
                return self
            else:
                return self._item['short_name']
    
        def name(self):
            return self._item['name']
        
    def __repr__(self):
        return str('<pycollector.project.Activity: activities=%d>' % len(self._itemdict))
    
    def __init__(self, scandict):
        self._itemdict = {k['activity_id']:k for k in scandict}

    def __getitem__(self, id):
        assert id in self._itemdict, "Unknown activity ID '%s'" % id
        return self._Activity(self._itemdict[id])

    def activitiesids(self):
        return set(self._itemdict.keys())
    
    def to_shortname(self, a):
        d = {v['name']:v['short_name'] for (k,v) in self._itemdict.items()}
        assert a in d, "Activity '%s' not found" % a
        return d[a]
        
    def labels(self):
        return set([v['name'] for v in self._itemdict.values()])

    def ids(self):
        return self.activitiesids()
    
    def dict(self):
        return self._itemdict
        
    def new(self, name, program_name, project_name, collection_name, short_name):
        assert isinstance(name, str)                
        assert isinstance(program_name, str)
        assert isinstance(project_name, str)
        assert isinstance(short_name, str)
        item = {'activity_id':str(uuid.uuid4()),
                'active':False,
                'project_name':project_name,
                'program_name':program_name,
                'collection_name':collection_name,
                'counts': 0,
                'created_date':timestamp(),
                'id':'_'.join([program_name, project_name, collection_name]),
                'name':name,
                'short_name':short_name,
                'updated_date':timestamp(),
                }

        co_Activity = pycollector.globals.backend().table.activity
        co_Activity.put_item(Item=item)
        self._itemdict[item['activity_id']] = item
        return item['activity_id']
    
    
class Collection(object):
    """ collector.project.Collections class
    
        An interface to the Collections table which defines all of each Collection() available to collectors. 
    """


    class _Collection(object):
        """collector.project.Collection

        A single collection definition, as specified in the Collections table.  
        """
    
        def __init__(self, itemdict):
            self._item = itemdict
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
            assert len(self.shortnames()) == len(self.activities())
        
        def __repr__(self):
            return str('<pycollector.project.Collection: "%s", activities=%d, project=%s>' % (self.name(), self.num_activities(), self.project()))

        def project(self):
            return self._item['project_name']

        def id(self):
            return self._item['collection_id']
    
        def name(self):
            return self._item['name']

        def activities(self):
            return self._item['activities'].split(',')

        def shortnames(self):
            return [x.lower() for x in self._item['activity_short_names'].split(',')]
    
        def num_activities(self):
            return len(self.activities())

        def shortname_to_activity(self, shortname):
            assert shortname.lower() in self.shortnames(), 'Shortname "%s" not in instance "%s"' % (shortname, str(self.shortnames()))    
            return self.activities()[self.shortnames().index(shortname.lower())]

        def activity_to_shortname(self, a):
            assert a.lower() in self.activities(), 'Activity "%s" not found in "%s"' % (a, str(self.activities()))
            return self.shortnames()[self.activities().index(a.lower())]
        
        def dict(self):
            return self._item
    
        def enable(self):
            self._item['active'] = True
            self._item['updated_date'] = timestamp()
            pycollector.globals.backend().table.collection.put_item(Item=self._item)
            return self

        def update(self, description, buttons, active):
            self._item['active'] = active
            self._item['collection_description'] = description
            self._item['activity_short_names'] = buttons
            self._item['updated_date'] = timestamp()
            pycollector.globals.backend().table.collection.put_item(Item=self._item)        
            return self
    
        def disable(self):
            self._item['active'] = False
            self._item['updated_date'] = timestamp()
            pycollector.globals.backend().table.collection.put_item(Item=self._item)
            return self

        def description(self, desc=None):
            if desc is None:
                return self._item['collection_description']
            else:
                assert isinstance(desc, str), "Description must be a string"
                self._item['collection_description'] = desc
                self._item['updated_date'] = timestamp()
                pycollector.globals.backend().table.collection.put_item(Item=self._item)
                return self

        def buttons(self, buttonlist):
            assert isinstance(buttonlist, list) and isinstance(buttonlist[0], str), "Buttons must be a list of strings"
            assert all([len(b.split(','))==1 for b in buttonlist]), "Button strings cannot contain commas"
            self._item['activity_short_names'] = ','.join(buttonlist)
            self._item['updated_date'] = timestamp()
            pycollector.globals.backend().table.collection.put_item(Item=self._item)
            return self

        def training_videos(self, urls=None, maxvideos=50):
            if urls is not None:
                new_urls = tolist(urls)
                current_urls = self._item['training_videos']
                assert all([isurl(url) for url in urls])            
                self._item['training_videos'] = [url for url in new_urls if url not in current_urls] + current_urls  # prepend
                if maxvideos is not None and len(self._item['training_videos']) > maxvideos:
                    self._item['training_videos'] = self._item['training_videos'][:maxvideos]  # remove older URLs
                self._item['updated_date'] = timestamp()
                self._item['isTrainingVideoEnabled'] = 'true'
                pycollector.globals.backend().table.collection.put_item(Item=self._item)        
                return self            
            else:
                return self._item['training_videos']
    
    def __init__(self, scandict):
        self._itemdict = {k['name']:k for k in scandict}

    def __repr__(self):
        return str('<collector.project.Collections: projects=%d, collections=%d>' % (len(groupbyasdict(self._itemdict.values(), lambda v: v['project_name'])), len(self._itemdict)))
    
    def __getitem__(self, name):
        key = name if name in self._itemdict else (self.id_to_name(name) if self.id_to_name(name) in self._itemdict else None)
        assert key is not None, "Unknown collection name '%s'" % name        
        return self._Collection(self._itemdict[key])
    
    def collectionids(self):
        return set([v['collection_id'] for v in self._itemdict.values()])

    def isvalid(self, name):
        try:
            self.__getitem__(name)
            return True
        except:
            return False

    def collection(self, name):
        return self.__getitem__(name)
    
    def id_to_name(self, id=None):
        d = {v.id():v.name() for v in self.collectionlist()}
        return d[id] if id is not None else d
    
    def names(self):
        return set([v['name'] for (k,v) in self._itemdict.items()])
    
    def keys(self):
        return self.names()
    
    def dict(self):
        return {v.name():v for v in self.collectionlist()}

    def collectionlist(self):
        return [self[k] for k in self.names()]

    def to_df(self):
        return pd.DataFrame([v for k,v in self._itemdict.items() ])
    
    def new(self, name, activities, program_name, project_name, description, buttons, objects, consent_overlay_text='Please select the record button, say "I consent to this video collection”', training_videos=None):
        """ Create new collections with activities and objects details

        Args:
            name ([string]): the name of the collection
            activities ([string]): comma seperated string for list of activities
            program_name ([type]): [description]
            project_name ([type]): [description]
            description ([type]): [description]
            buttons ([string]): comma seperated string for list of activities with CAP
            objects ([type]): [description]
            training_videos ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        (activities, buttons) = (tolist(activities), tolist(buttons))
        assert isinstance(name, str)                
        assert isinstance(activities, list) and all([isinstance(a, str) for a in activities])
        assert isinstance(program_name, str)
        assert isinstance(project_name, str)
        assert isinstance(description, str)
        assert isinstance(buttons, list) and all([isinstance(b, str) for b in buttons]) and len(buttons) == len(activities)

        # Add new program (if not present)
        response = pycollector.globals.backend().table.program.query(KeyConditionExpression=Key("id").eq(program_name))
        if not any([x['name'] == program_name for x in response['Items']]):
            item = {'id':program_name,
                    'name':project_name,
                    'client':program_name,
                    'created_date':timestamp(),
                    'program_id':str(uuid.uuid4())}
            pycollector.globals.backend().table.program.put_item(Item=item)            
            
        # Add new project (if not present)
        response = pycollector.globals.backend().table.project.query(KeyConditionExpression=Key("id").eq(program_name))
        if not any([x['name'] == project_name for x in response['Items']]):
            item = {'id':program_name,
                    'name':project_name,
                    'created_date':timestamp(),
                    'mobile_id':project_name,
                    'project_id':str(uuid.uuid4())}
            pycollector.globals.backend().table.project.put_item(Item=item)            
                                     
        # Add activities:
        #   Assumed the order of activities and buttons are the same. And they are one to one mapped.
        for idx, activity in enumerate(activities):
            a = pycollector.globals.backend().activity()
            newid = a.new(name=activity, program_name=program_name, project_name=project_name, collection_name=name, short_name=buttons[idx])
            a[newid].enable()
            
        item = {'collection_id':str(uuid.uuid4()),
                'activities':','.join(activities),
                'active':False,
                'project_name':project_name,
                'program_name':program_name,
                'created_date':timestamp(),
                'id':'_'.join([program_name, project_name]),
                'name':name,
                'default_object': objects.split(',')[0],
                'objects_list': objects,
                'collection_description':description,
                'activity_short_names':','.join(buttons),
                'updated_date':timestamp(),
                'training_videos':training_videos if training_videos is not None else [],
                'training_videos_low':[],
                'isTrainingVideoEnabled': True,
                'isConsentRequired': True,
                'consent_overlay_text': consent_overlay_text,
                }

        pycollector.globals.backend().table.collection.put_item(Item=item)
        self._itemdict[item['collection_id']] = item
        return item['collection_id']

    
class Rating(object):
    """collector.project.Rating() class

       An interface to the ratings table.
    """
    def __init__(self, ratingdict=None):
        """ratingdict is a single row of the table"""
        assert isinstance(ratingdict, dict)
        self._item = {k.lower():v for (k,v) in ratingdict.items()}
        #assert 'up' in self._item or 'good' in self._item

    def reviewer(self):
        return self._item['reviewer_id']

    def review_score(self):
        return 1.0 if (('up' in self._item and self._item['up'] > 0) or self._item['good_for_training']>0 or ('good' in self._item and self._item['good'] > 0)) else 0.0

    def isperfect(self):
        return 'good_for_training' in self._item and self._item['good_for_training'] > 0

    def is_processed(self):
        """Has a rating been processed by the lambda function yet?"""
        return 'up' in self._item or 'good' in self._item and (len(self._item['rating_responses'])>0 and len(self._item['rating_responses'][0])>0)
        
    def isgood(self):
        return ('up' in self._item and self._item['up'] > 0) or ('good' in self._item and self._item['good'] > 0)

    def instanceid(self):
        return self._item['id']
    
    def is_repeated_scene(self):
        return 'bad_diversity' in self._item and self._item['bad_diversity'] > 0

    def is_awkward(self):
        return ('bad_scene' in self._item and self._item['bad_scene'] > 0) or ('awkward_scene' in self._item and self._item['awkward_scene'] > 0)
    
    def is_bad_viewpoint(self):
        return 'bad_viewpoint' in self._item and self._item['bad_viewpoint'] > 0
    
    def review_reason(self):
        bad_desc = ["Incorrect label" if ('bad_label' in self._item and self._item['bad_label'] > 0) else "",
                    "Box too big" if ('bad_box_big' in self._item and self._item['bad_box_big'] > 0) else "",
                    "Box too small" if ('bad_box_small' in self._item and self._item['bad_box_small'] > 0) else "",
                    "Incorrect timing" if ('bad_timing' in self._item and self._item['bad_timing'] > 0) else "",
                    "Box not centered" if ('bad_alignment' in self._item and self._item['bad_alignment'] > 0) else "",
                    "Object/activity not visible" if ('bad_visibility' in self._item and self._item['bad_visibility'] > 0) else "",
                    "Unusable video" if ('bad_video' in self._item and self._item['bad_video'] > 0) else ""]
        bad_desc = [d for d in bad_desc if len(d) > 0]

        warn_desc = ["Incorrect viewpoint" if ('bad_viewpoint' in self._item and self._item['bad_viewpoint'] > 0) else "",
                        "Repeated scene" if ('bad_diversity' in self._item and self._item['bad_diversity'] > 0) else "",
                        "Awkward scene" if (('bad_scene' in self._item and self._item['bad_scene'] > 0) or ('awkward_scene' in self._item and self._item['awkward_scene'] > 0)) else ""]
        warn_desc = [d for d in warn_desc if len(d) > 0]
        
        good_desc = ['Good' if ((('up' in self._item and self._item['up'] > 0) or ('good' in self._item and self._item['good'] > 0)) and self._item['good_for_training'] == 0) else 'Perfect' if ('good_for_training' in self._item and self._item['good_for_training']>0) else '']
        good_desc = [d for d in good_desc if len(d) > 0]

        assert not ((len(good_desc)>0) and (len(bad_desc)>0)), "Invalid review_reason for instance id %s" % self.instanceid()
        desc = good_desc + bad_desc + warn_desc
        return desc

    def updated(self):
        assert 'updated_time' in self._item, "'updated_time' not present in '%s'" % (str(self._item))
        try:
            return collector.util.fromtimestamp(self._item['updated_time'])  
        except:
            try:                
                return datetime.strptime(self._item['updated_time'], "%Y-%m-%dT%H:%M:%S%z")  # 2020-08-10T22:47:53-04:00 format
            except:
                et = pytz.timezone("US/Eastern")            
                return datetime.strptime(self._item['updated_time'], "%m/%d/%Y, %H:%M:%S %p").astimezone(et)  # HACK to fix Heng's timestamp bug

            
class CollectionInstance(object):
    """collector.project.CollectionInstance class

       A CollectionInstance() is an observed Collection() made up of one or more Instance() of a specified Activity()
    """
    def __init__(self, collection_name, video_id, collector, table=None):
        self._collection_name = collection_name
        self._video_id = video_id
        self._collector = collector
        self._table = table

    def __repr__(self):
        return str('<collector.project.CollectionInstance: collection="%s", collector=%s, videoid=%s, uploaded=%s>' % (self._collection_name, self.collector(), self._video_id, self.uploaded()))

    def collector(self):
        return self._collector
    
    def video(self):
        return Video(self._video_id, attributes=self._table)

    def has_rating(self):
        return self._table is not None and 'Unrated' not in self.review_reason()

    def is_good(self, t=0.5):
        return self._table is not None and 'rating_score' in self._table and self._table['rating_score'] > t

    def is_bad_viewpoint(self, t=0.5):
        return self._table is not None and 'bad_viewpoint_score' in self._table and self._table['bad_viewpoint_score'] > t

    def is_repeated_scene(self, t=0.5):
        return self._table is not None and 'bad_diversity_score' in self._table and self._table['bad_diversity_score'] > t

    def is_awkward(self):
        return self._table is not None and ('bad_scene' in self._table and self._table['bad_scene_score'] > 0) or ('awkward_scene' in self._table and self._table['awkward_scene_score'] > 0)
    
    def uploaded(self):
        assert self._table is not None and 'collected_date' in self._table
        return self._table['collected_date']

    def thumbnail(self):
        assert self._table is not None and 'thumbnail' in self._table        
        return vipy.image.Image(url=self._table['thumbnail'])    
        
    def review_reason(self):
        #assert (self._table is not None and
        #        'bad_label_score' in self._table and
        #        'bad_box_big_score' in self._table and
        #        'bad_box_small_score' in self._table and
        #        'bad_viewpoint_score' in self._table and
        #        'bad_timing_score' in self._table and
        #        'bad_alignment_score' in self._table and
        #        'bad_visibility_score' in self._table and
        #        'bad_diversity_score' in self._table and
        #        'bad_video_score' in self._table and
        #        'awkward_scene_score' in self._table and                
        #        'rating_score' in self._table)
        
        desc = ["label" if ('bad_label_score' in self._table and self._table['bad_label_score'] > 0) else "",
                "box (too big)" if ('bad_box_big_score' in self._table and self._table['bad_box_big_score'] > 0) else "",
                "box (too small)" if ('bad_box_small_score' in self._table and self._table['bad_box_small_score'] > 0) else "",
                "viewpoint" if ('bad_viewpoint_score' in self._table and self._table['bad_viewpoint_score'] > 0) else "",
                "timing" if ('bad_timing_score' in self._table and self._table['bad_timing_score'] > 0) else "",
                "box (not centered)" if ('bad_alignment_score' in self._table and self._table['bad_alignment_score'] > 0) else "",
                "object/activity (not visible)" if ('bad_visibility_score' in self._table and self._table['bad_visibility_score'] > 0) else "",
                "scene (repeated)" if ('bad_diversity_score' in self._table and self._table['bad_diversity_score'] > 0) else "",
                "awkward scene" if ('awkward_scene_score' in self._table and self._table['awkward_scene_score'] > 0) else "",                
                "video content" if ('bad_video_score' in self._table and self._table['bad_video_score'] > 0) else ""]
        
        desc = [d for d in desc if len(d) > 0]
        desc = "incorrect " + " and ".join(desc) if len(desc) > 0 else 'Unrated' if ('rating_score' in self._table and self._table['rating_score'] == 0) else 'Good'
        return desc.rstrip()

    def review_score(self):
        return self._table['rating_score']

    def name(self):
        return self._collection_name

    def videoid(self):
        return self._video_id

    def subjectid(self):
        assert 'subject_id' in self._table and len(self._table['subject_id']) == 1
        return self._table['subject_id'][0]

    def collectorid(self):
        assert 'collector_id' in self._table
        return self._table['collector_id']

    
        
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
        
        print("[pycollector.project]:  Returned %d videos" % len(self.df))

    def __repr__(self):
        return str("<collector.project: program=%s, videos=%d, since=%s, collectors=%d>" % (self._programid,
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
                r["collecgtor_id" if isapi('v1') else "collector_email"],
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


def search():
    """Return all videos for a python user"""

    # this should:
    # ask the user to login, triggered by accessing pycollector.backend.Backend() without credentials
    # If the user has previously logged in, we save a credentials.pkl file containing their access token so that this does not need login the next time
    # query the video table for this user using the pycollector.video.Project() tools
    # Return a list of pycollector.video.Video() objects
    
    pass
    
