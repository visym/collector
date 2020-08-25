import os
import boto3
import getpass
from pycollector.util import is_email_address



class Backend(object):
    """Standard interface for project administration on Collector backend

        User will need to set up their local environment variables         
    """

    def __init__(self, region=None, verbose=True):
        self._region = region
        self._verbose = verbose

        # TODO - Will add to conditional checks on initialize the backend. Which also help to fail gracefully.

        # Check if running local with environment variables
        if "VISYM_COLLECTOR_AWS_ACCESS_KEY_ID" in os.environ:
            self._s3_client = boto3.client(
                "s3",
                region_name=os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"],
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ[
                    "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"
                ],
            )

            self._dynamodb_client = boto3.client(
                "dynamodb",
                region_name=os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"],
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ[
                    "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"
                ],
            )

            self._dynamodb_resource = boto3.resource(
                "dynamodb",
                region_name=os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"],
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ[
                    "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"
                ],
            )

            self._cognitoidP_client = boto3.client(
                "cognito-idp",
                region_name=os.environ["VISYM_COLLECTOR_AWS_REGION_NAME"],
                aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ[
                    "VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"
                ],
            )

            self._cognitoUserPoolid = os.environ[
                "VISYM_COLLECTOR_AWS_COGNITO_USER_POOL_ID"
            ]
            self._cognitoAppClientlid = os.environ[
                "VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_ID"
            ]
            self._cognitoAppClientlSecret = os.environ[
                "VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_SECRET"
            ]
        # Else if running on AWS Lambda or using AWS CLI config
        else:
            self._s3_client = boto3.client("s3")
            self._dynamodb_client = boto3.client("dynamodb")
            self._dynamodb_resource = boto3.resource("dynamodb")
            self._cognitoidP_client = boto3.client("cognito-idp")

            
    def login(self, email):
        assert is_email_address(email)
        password = getpass.getpass()
        
        

class Test(object):
    def __init__(self):
        pass

class Dev(object):
    def __init__(self):
        pass

class Prod_v1(Backend):
    def __init__(self):
        super(Prod_v1, self).__init__()
        self.instances = self._dynamodb_resource.Table("co_Instances_dev")
        self.rating = self._dynamodb_resource.Table("co_Rating")
        self.program = self._dynamodb_resource.Table("co_Programs")
        self.video = self._dynamodb_resource.Table('co_Videos')    
        
    
class Prod(Backend):
    def __init__(self):
        super(Prod, self).__init__()        
        self.s3_bucket = 'diva-prod-data-lake174516-visym'
        self.video = self._dynamodb_resource.Table('strVideos-hirn6lrwxfcrvl65xnxdejvftm-visym')
        self.instances = self._dynamodb_resource.Table("strInstances-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self.rating = self._dynamodb_resource.Table("strRating-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self.program = self._dynamodb_resource.Table("strProgram-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self.collection = self._dynamodb_resource.Table("strCollections-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self.projects = self._dynamodb_resource.Table("strProjects-hirn6lrwxfcrvl65xnxdejvftm-visym")    
        self.activity = self._dynamodb_resource.Table("strActivities-hirn6lrwxfcrvl65xnxdejvftm-visym")
            
