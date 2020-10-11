import os
import boto3
import botocore
from botocore.config import Config
from datetime import datetime, timedelta
import decimal
import pandas as pd
import numpy as np
#import logging
import pycollector
import getpass
from pycollector.globals import print, GLOBALS
from pycollector.util import is_email_address


class User(object):
    """User class for collector's user management
    """

    def __init__(self, username=None, password=None):
        """
       
        Args:
            username ([type]): [description].
            password ([type], optional): [description]. Defaults to None.
        """

        self.app_client_id = GLOBALS['COGNITO']['app_client_id']
        self.identity_pool_id = GLOBALS['COGNITO']['identity_pool_id']
        self.provider_name = GLOBALS['COGNITO']['provider_name']
        self.region_name = GLOBALS['COGNITO']['region_name']
        
        # Initialize the base user properties and create the logger used by the function
        self._is_login = False
        self._token_initialized_time = None
        self._token_expiration_time = None 
        self._program_name = None

        # Initialize cognito clients
        # Ensure the system AWS credentials are not being used
        config = Config(signature_version=botocore.UNSIGNED)
        self._cognito_idp_client = boto3.client('cognito-idp', config=config, region_name=self.region_name)
        self._cognito_id_client = boto3.client('cognito-identity', config=config, region_name=self.region_name)

        # Login
        if username is None and 'VISYM_COLLECTOR_EMAIL' in os.environ:
            username = os.environ['VISYM_COLLECTOR_EMAIL']
        self._username = username
        if password is not None:
            self.login(password)

        self.refresh()


    def refresh(self):
        if 'VIPY_AWS_SESSION_TOKEN' in os.environ:
            self._set_S3_clients()
            self._set_lambda_clients()
            self._is_login = True
            self._cognito_username = os.environ['VIPY_AWS_COGNITO_USERNAME']
        return self
        
    def login(self, password=None):
        """[summary]

        Args:
            username ([type]): [description]
            password ([type]): [description]
        """
        
        username = self._username if self._username is not None else input("Collector email: ")
        assert is_email_address(username), 'Invalid collector email address "%s"' % username
        password = password if password is not None else getpass.getpass()
            
        try:
            signin_response = self._cognito_idp_client.initiate_auth(
                ClientId=self.app_client_id,
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
                IdentityPoolId=self.identity_pool_id,
                Logins={self.provider_name: self._id_token}
            )
            self._identity_id = get_id_response['IdentityId']

            # Get user temp AWS credentials
            get_aws_credentials_response = self._cognito_id_client.get_credentials_for_identity(
                IdentityId=self._identity_id,
                Logins={self.provider_name: self._id_token},
            )
            self._aws_credentials =  get_aws_credentials_response['Credentials']

            # Set up AWS services
            self._set_os_environ()            
            self._set_S3_clients()
            self._set_lambda_clients()
            self._is_login = True
            

        except Exception as e:
            custom_error = 'Failed to sign in due to exception: {0}'.format(e)
            raise Exception(custom_error)

        return self

    def _set_S3_clients(self):
        """[summary]
        """
        assert 'VIPY_AWS_SESSION_TOKEN' in os.environ        
        self._s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['VIPY_AWS_SESSION_TOKEN'],
            region_name=self.region_name
        )
        self._s3_resource = boto3.resource(
            's3',
            aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['VIPY_AWS_SESSION_TOKEN'],
            region_name=self.region_name
        )

    def _set_lambda_clients(self):
        """[summary]
        """
        assert 'VIPY_AWS_SESSION_TOKEN' in os.environ
        self._lambda_client = boto3.client(
            'lambda',
            aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ['VIPY_AWS_SESSION_TOKEN'],
            region_name=self.region_name
        )

    def _set_os_environ(self):
        """[summary]
        """  
        os.environ['VIPY_AWS_ACCESS_KEY_ID'] = self._aws_credentials['AccessKeyId']
        os.environ['VIPY_AWS_SECRET_ACCESS_KEY'] = self._aws_credentials['SecretKey']
        os.environ['VIPY_AWS_SESSION_TOKEN'] = self._aws_credentials['SessionToken']
        os.environ['VIPY_AWS_COGNITO_USERNAME'] = self._cognito_username
        
    def is_token_expired(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._token_expiration_time is None or datetime.now() > self._token_expiration_time 

    def token_expired_by(self):
        return self._token_expiration_time
    
    def is_authenticated(self):
        return self._is_login
    

    def add_user_to_group(self):
        """Check if the current user is already in the pycollector user group, if not add the user to group

        Returns:
            [type]: [description]
        """
    
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



    # self._cognitoidP_client = boto3.client(
    #     "cognito-idp",
    #     region_name=self._region,
    #     aws_access_key_id=os.environ["VISYM_COLLECTOR_AWS_ACCESS_KEY_ID"],
    #     aws_secret_access_key=os.environ["VISYM_COLLECTOR_AWS_SECRET_ACCESS_KEY"],
    # )

    # self._cognitoUserPoolid = os.environ["VISYM_COLLECTOR_AWS_COGNITO_USER_POOL_ID"]
    # self._cognitoAppClientlid = os.environ["VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_ID"]
    # self._cognitoAppClientlSecret = os.environ["VISYM_COLLECTOR_AWS_COGNITO_APP_CLIENT_SECRET"]
