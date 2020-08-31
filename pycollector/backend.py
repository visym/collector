import os
import boto3
import getpass
from pycollector.util import is_email_address
from vipy.globals import print


class Backend(object):
    """Standard interface for project administration on Collector backend

        User will need to set up their local environment variables         
    """

    def __init__(self, region='us-east-1', verbose=True, cache=True):
        self._region = os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"] if 'VISYM_COLLECTOR_AWS_REGION_NAME' in os.environ else region
        self._verbose = verbose
        self._cache = cache

        self._s3_bucket = None
        self._ddb_video = None
        self._ddb_instance = None
        self._ddb_rating = None
        self._ddb_program = None
        self._ddb_collection = None
        self._ddb_project = None
        self._ddb_activity = None
                
        self._program = None
        self._collection = None
        self._activity = None

        
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
        import pycollector.project  # avoid circular import
        self._program = pycollector.project.Program(self._ddb_program.scan()['Items']) if (self._program is None or self._cache is False) else self._program
        return self._program
        
    def activity(self):
        import pycollector.project  # avoid circular imports
        self._activity = pycollector.project.Activity(self._ddb_activity.scan()['Items']) if (self._activity is None or self._cache is False) else self._activity
        return self._activity
        
    def collection(self):
        import pycollector.project  # avoid circular imports
        self._collection = pycollector.project.Collection(self._ddb_collection.scan()['Items']) if (self._collection is None or self._cache is False) else self._collection
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

    
class Test(Backend):
    def __init__(self):
        super(Test, self).__init__()        
        self._s3_bucket = 'diva-prod-data-lake232615-visymtest'
        self._ddb_video = self._dynamodb_resource.Table('strVideos-uxt26i4hcjb3zg4zth4uaed4cy-visymtest')
        self._ddb_instance = self._dynamodb_resource.Table("strInstances-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_rating = self._dynamodb_resource.Table("strRating-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_program = self._dynamodb_resource.Table("strProgram-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_collection = self._dynamodb_resource.Table("strCollections-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_project = self._dynamodb_resource.Table("strProjects-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")    
        self._ddb_activity = self._dynamodb_resource.Table("strActivities-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_subject = self._dynamodb_resource.Table("strSubject-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_collector = self._dynamodb_resource.Table("strCollector-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")

        
class Dev(object):
    def __init__(self):
        pass

    
class Prod_v1(Backend):
    def __init__(self):
        super(Prod_v1, self).__init__()
        self._ddb_instance = self._dynamodb_resource.Table("co_Instances_dev")
        self._ddb_rating = self._dynamodb_resource.Table("co_Rating")
        self._ddb_program = self._dynamodb_resource.Table("co_Programs")
        self._ddb_video = self._dynamodb_resource.Table('co_Videos')    
        
    
class Prod(Backend):
    def __init__(self):
        super(Prod, self).__init__()        
        self._s3_bucket = 'diva-prod-data-lake174516-visym'
        self._ddb_video = self._dynamodb_resource.Table('strVideos-hirn6lrwxfcrvl65xnxdejvftm-visym')
        self._ddb_instance = self._dynamodb_resource.Table("strInstances-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_rating = self._dynamodb_resource.Table("strRating-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_program = self._dynamodb_resource.Table("strProgram-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_collection = self._dynamodb_resource.Table("strCollections-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_project = self._dynamodb_resource.Table("strProjects-hirn6lrwxfcrvl65xnxdejvftm-visym")    
        self._ddb_activity = self._dynamodb_resource.Table("strActivities-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_subject = self._dynamodb_resource.Table("strSubject-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_collector = self._dynamodb_resource.Table("strCollector-hirn6lrwxfcrvl65xnxdejvftm-visym")
            
