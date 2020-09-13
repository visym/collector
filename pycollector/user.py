import os
import boto3
import botocore
from botocore.config import Config
from datetime import datetime, timedelta
import decimal
import pandas as pd
import numpy as np
import logging


# TODO - Put these public variables somewhere
app_client_id = '6k20qruljfs0v7n5tmt1pk0u1q' # 
identity_pool_id = 'us-east-1:c7bbbc40-37d3-4ad8-8afd-492c095729bb'
provider_name = 'cognito-idp.us-east-1.amazonaws.com/us-east-1_sFpJQRLiY'

class User(object):
    """User class for collector's user management
    """

    def __init__(self, username=None, password=None):
        """
        

        Args:
            username ([type], optional): [description]. Defaults to None.
            password ([type], optional): [description]. Defaults to None.
        """

        #####################################################################
        # Initialize the base user properties and create the logger used by the function
        #####################################################################
        self._is_login = False
        self._token_initialized_time = None
        self._token_expiration_time = None 
        self._program_name = None
        self._logger = logging.getLogger(User.__name__)

        #####################################################################
        # Initialize cognito clients
        #####################################################################
        # Ensure the system AWS credentials are not being used
        config = Config(signature_version=botocore.UNSIGNED)
        self._cognito_idp_client = boto3.client('cognito-idp', config=config)
        self._cognito_id_client = boto3.client('cognito-identity', config=config)
        # self._cognito_idp_client = boto3.client('cognito-idp')
        # self._cognito_id_client = boto3.client('cognito-identity')

        #####################################################################
        # Login  
        #####################################################################
        if username and password:
            self.login(username, password)

        #####################################################################
        # Term of usages properties 
        #####################################################################



    def login(self, username, password):
        """[summary]

        Args:
            username ([type]): [description]
            password ([type]): [description]
        """

        try:
            signin_response = self._cognito_idp_client.initiate_auth(
                ClientId=app_client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={'USERNAME': username, 'PASSWORD': password })

            # Set user properties
            self._username = username


            # Should we expose these tokens?
            self._access_token = signin_response['AuthenticationResult']['AccessToken']
            self._id_token = signin_response['AuthenticationResult']['IdToken']

            # Get cognito username 
            getuser_response = self._cognito_idp_client.get_user(AccessToken=self._access_token )
            self._cognito_username = getuser_response['Username']
    
            # Set token expiration time
            token_expires_in_secs = signin_response['AuthenticationResult']['ExpiresIn']    
            self._token_initialized_time = datetime.now()
            self._token_expiration_time = datetime.now() + timedelta(0,token_expires_in_secs) 

            # Get user identity_id
            get_id_response= self._cognito_id_client.get_id(
                IdentityPoolId=identity_pool_id,
                Logins={provider_name: self._id_token}
            )
            self._identity_id = get_id_response['IdentityId']

            # Get user temp AWS credentials
            get_aws_credentials_response = self._cognito_id_client.get_credentials_for_identity(
                IdentityId=self._identity_id,
                Logins={provider_name: self._id_token},
            )
            self._aws_credentials =  get_aws_credentials_response['Credentials']

            # Set up AWS services 
            self.set_S3_clients()
            self.set_lambda_clients()


        except Exception as e:
            custom_error = 'Failed to sign in due to exception: {0}'.format(e)
            raise Exception(custom_error)


    def set_S3_clients(self):
        """[summary]
        """
        self._s3_client = boto3.client(
            's3',
            aws_access_key_id=self._aws_credentials['AccessKeyId'],
            aws_secret_access_key=self._aws_credentials['SecretKey'],
            aws_session_token=self._aws_credentials['SessionToken'],
        )
        self._s3_resource = boto3.resource(
            's3',
            aws_access_key_id=self._aws_credentials['AccessKeyId'],
            aws_secret_access_key=self._aws_credentials['SecretKey'],
            aws_session_token=self._aws_credentials['SessionToken'],
        )

    def set_lambda_clients(self):
        """[summary]
        """
        self._lambda_client = boto3.client(
            'lambda',
            aws_access_key_id=self._aws_credentials['AccessKeyId'],
            aws_secret_access_key=self._aws_credentials['SecretKey'],
            aws_session_token=self._aws_credentials['SessionToken'],
        )

        # self._lambda_client = boto3.client(
        #     'lambda',
        # )

    def new_collection(self, name, organization_name, program_name,project_name, description, activities, activity_short_names, objects ):

        # Invoke Lambda function
        request = {'identity_id': identity_id, 'cognito_username': cognito_username, 'email': email}


        # Invoke Lambda function
        try:
            response = self._lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:806596299222:function:CollectorTestPostAuthentication',
            InvocationType= 'RequestResponse',
            LogType='Tail',
            Payload=json.dumps(request),
        )
        except Exception as e:
            custom_error = '\nException : failed to invoke jobs.\n'
            custom_error += 'Error : ' + str(e) + '\n'
            raise Exception(custom_error)

        
    def is_token_expired(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return datetime.now() > self._token_expiration_time 

    def token_expired_by(self):
        return self._token_expiration_time

    @staticmethod
    def static_method():
        ...

    @property
    def username(self):
        """
        """
        return self._username
    @property
    def cognito_username(self):
        """
        """
        return self._cognito_username

    @property
    def lambda_client(self):
        """
        """
        return self._lambda_client

