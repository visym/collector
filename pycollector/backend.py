import os
import boto3
import getpass
from pycollector.util import is_email_address
from vipy.globals import print
from vipy.util import groupbyasdict


class Backend(object):
    """Standard interface for project administration on Collector backend

        User will need to set up their local environment variables         
    """

    def __init__(self, region='us-east-1', verbose=True, cache=True):
        self._region = os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"] if 'VISYM_COLLECTOR_AWS_REGION_NAME' in os.environ else region
        self._verbose = verbose
        self._cache = cache

        self._program = None
        self._project = None
        self._collection = None
        self._activity = None

        # Overloaded by specific backend
        self._s3_bucket = None
        self._ddb_video = None
        self._ddb_instance = None
        self._ddb_rating = None
        self._ddb_program = None
        self._ddb_collection = None
        self._ddb_project = None
        self._ddb_activity = None
                
        
        # TODO - Will add to conditional checks on initialize the backend. Which also help to fail gracefully.

        # Check if running local with environment variables
        if "VISYM_COLLECTOR_AWS_ACCESS_KEY_ID" in os.environ:
            self._s3_client = boto3.client(
                "s3",
                region_name=self._region,
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"],
            )

            self._dynamodb_client = boto3.client(
                "dynamodb",
                region_name=self._region,
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"],
            )

            self._dynamodb_resource = boto3.resource(
                "dynamodb",
                region_name=self._region,
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"],
            )

            self._cognitoidP_client = boto3.client(
                "cognito-idp",
                region_name=self._region,
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"],
            )

            self._cognitoUserPoolid = os.environ["VISYM_COLLECTOR_AWS_COGNITO_USER_POOL_ID"]
            self._cognitoAppClientlid = os.environ["VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_ID"]
            self._cognitoAppClientlSecret = os.environ["VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_SECRET"]
            

        else:
            # Else if running on AWS Lambda or using AWS CLI config            
            self._s3_client = boto3.client("s3")
            self._dynamodb_client = boto3.client("dynamodb")
            self._dynamodb_resource = boto3.resource("dynamodb")
            self._cognitoidP_client = boto3.client("cognito-idp")

            
    def login(self, email):
        assert is_email_address(email)
        password = getpass.getpass()


    def label(self):
        pass
        

    def program(self):
        self._program = Program(self._ddb_program.scan()['Items']) if (self._program is None or self._cache is False) else self._program
        return self._program

    def project(self):
        self._project = Project(self._ddb_project.scan()['Items']) if (self._project is None or self._cache is False) else self._project
        return self._project
        
    def activity(self):
        self._activity = Activity(self._ddb_activity.scan()['Items']) if (self._activity is None or self._cache is False) else self._activity
        return self._activity
        
    def collection(self):
        self._collection = Collection(self._ddb_collection.scan()['Items']) if (self._collection is None or self._cache is False) else self._collection
        return self._collection
        
    def __getattr__(self, name):
        if name == 'table':
            # For dotted attribute access to named DDB tables
            class _PyCollector_Backend_Tables(object):
                def __init__(self, program, project, collection, activity, video, instance, rating, subject, collector):
                    self.program = program
                    self.project = project
                    self.collection = collection
                    self.activity = activity                    
                    self.video = video
                    self.instance = instance
                    self.rating = rating
                    self.subject = subject
                    self.collector = collector

            return _PyCollector_Backend_Tables(self._ddb_program,
                                               self._ddb_project,
                                               self._ddb_collection,
                                               self._ddb_activity,  
                                               self._ddb_video,                                               
                                               self._ddb_instance,
                                               self._ddb_rating,
                                               self._ddb_subject,
                                               self._ddb_collector)
        else:
            return self.__getattribute__(name)

    def s3_bucket(self):
        return self._s3_bucket

    
            
class CollectorAssignment(object):
    """ class for collector assignment

    Args:
        object ([type]): [description]
    """

    def __init__(self, program_name=None, project_name=None):
        
        self.program_name = program_name
        self.project_name = project_name

        # Get Program information
        self.program = pycollector.globals.backend().program()[self.program_name]
        # Get Project information
        self.project = pycollector.globals.backend().project()

        # Set program and project dataframe
        program_df = self.program.to_df()
        project_df = self.project.get_all_projects_in_df()

        self._program_df = program_df[['name','program_id']]
        self._project_df = project_df[['project_id','name']]

        self._program_df.rename(columns={"name" : "program_name"}, inplace=True)
        self._project_df.rename(columns={"name": "project_name"}, inplace=True)

        # Get Collections
        self._collections = pycollector.globals.backend().collection()
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



class Project(object):
    """ collector.backend.Project

        An interface to the Projects table
    """
        
    class _Project(object):
        """ A single project definition, as specified in the Projects table.
        """
        
        def __init__(self, itemdict):
            self._item = itemdict
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
            
        def __repr__(self):
            return str('<collector.backend.Project: "%s", name=%s, project_id=%s,  created_date=%s>' % (self._item['name'], self._item['name'], self._item['project_id'], self._item['created_date']))

        def dict(self):
            return self._item

        def to_df(self):
            return pd.DataFrame([self._item])

    def __init__(self, tabledict):
        self._itemdict = {k['name']:k for k in tabledict}
    
    def dict(self):
        return self._itemdict

    def __getitem__(self, name):
        assert name in self._itemdict, "Unknown project name '%s'" % name
        return self._Project(self._itemdict[name])

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
        co_Project = pycollector.globals.backend().table.project
        response = co_Project.query(KeyConditionExpression=Key("id").eq(program_name))
        if not any([x['name'] == project_name for x in response['Items']]):
            item = {'id':program_name,
                    'name':project_name,
                    'created_date':timestamp(),
                    'mobile_id':project_name,
                    'project_id':str(uuid.uuid4())}
            co_Project.put_item(Item=item)    
            print("Created New Project: ", project_name)
        else:
            print("This project %s is already exists. " % name)    

            


class Program(object):
    """ collector.backend.Programs

        An interface to the Programs table
    """

    class _Program(object):
        """A single program definition"""
    
        def __init__(self, itemdict):
            self._item = itemdict  #[k for k in co_Program.scan()["Items"] if k['name'] == name][0] if name is not None else item
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
        
        def __repr__(self):
            return str('<pycollector.backend.Program: "%s", name=%s, program_id=%s, client=%s, created_date=%s>' % (self._item['name'], self._item['name'], self._item['program_id'], self._item['client'], self._item['created_date']))

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
    """ collector.backend.Activity class

        An interface to the Activity table which defines the relationship between collections and activities.
    """


    class _Activity(object):
        """A single activity definition, as specified in the Activities table.
        """
    
        def __init__(self, itemdict):
            self._item = itemdict
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
        
        def __repr__(self):
            return str('<pycollector.backend.Activity: "%s", shortname=%s, id=%s, project=%s, collection=%s>' % (self._item['name'], self._item['short_name'], self._item['activity_id'], self._item['project_name'], self._item['collection_name']))

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
        return str('<pycollector.backend.Activity: activities=%d>' % len(self._itemdict))
    
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
    """ collector.backend.Collection class
    
        An interface to the Collections table which defines all of each Collection() available to collectors. 
    """


    class _Collection(object):
        """collector.backend.Collection

        A single collection definition, as specified in the Collections table.  
        """
    
        def __init__(self, itemdict):
            self._item = itemdict
            assert isinstance(self._item, dict) and 'name' in self._item, "invalid item"
            assert len(self.shortnames()) == len(self.activities())
        
        def __repr__(self):
            return str('<pycollector.backend.Collection: "%s", activities=%d, project=%s>' % (self.name(), self.num_activities(), self.project()))

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
        return str('<collector.backend.Collections: projects=%d, collections=%d>' % (len(groupbyasdict(self._itemdict.values(), lambda v: v['project_name'])), len(self._itemdict)))
    
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
    """collector.backend.Rating() class

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
            return fromtimestamp(self._item['updated_time'])  
        except:
            try:                
                return datetime.strptime(self._item['updated_time'], "%Y-%m-%dT%H:%M:%S%z")  # 2020-08-10T22:47:53-04:00 format
            except:
                et = pytz.timezone("US/Eastern")            
                return datetime.strptime(self._item['updated_time'], "%m/%d/%Y, %H:%M:%S %p").astimezone(et)  # HACK to fix Heng's timestamp bug

            
class CollectionInstance(object):
    """collector.backend.CollectionInstance class

       A CollectionInstance() is an observed Collection() made up of one or more Instance() of a specified Activity()
    """
    def __init__(self, collection_name, video_id, collector, table=None):
        self._collection_name = collection_name
        self._video_id = video_id
        self._collector = collector
        self._table = table

    def __repr__(self):
        return str('<pycollector.backend.CollectionInstance: collection="%s", collector=%s, videoid=%s, uploaded=%s>' % (self._collection_name, self.collector(), self._video_id, self.uploaded()))

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

    
        
