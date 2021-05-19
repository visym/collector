import os
import boto3
import botocore
from botocore.config import Config
import requests
from datetime import datetime, timedelta
import decimal
import pandas as pd
import numpy as np
import json

# import logging
import pycollector
import getpass
from pycollector.globals import GLOBALS  # print
from pycollector.util import is_email_address


class User(object):
    """User class for collector's user management"""

    def __init__(self, username=None, password=None):
        """

        Args:
            username ([str]): username of pycollector.
            password ([str], optional): password for the user. Defaults to None.
        """
        # TODO - Apply paramter store

        # self.app_client_id = GLOBALS["COGNITO"]["app_client_id"]
        # self.identity_pool_id = GLOBALS["COGNITO"]["identity_pool_id"]
        # self.provider_name = GLOBALS["COGNITO"]["provider_name"]
        # self.region_name = GLOBALS["COGNITO"]["region_name"]

        # Initialize the base user properties and create the logger used by the function
        self._is_login = False
        # self._token_initialized_time = None
        self._token_expiration_time = None
        self._program_name = None

        # # Initialize cognito clients
        # # Ensure the system AWS credentials are not being used
        # config = Config(signature_version=botocore.UNSIGNED)
        # self._cognito_idp_client = boto3.client("cognito-idp", config=config, region_name=self.region_name)
        # self._cognito_id_client = boto3.client("cognito-identity", config=config, region_name=self.region_name)

        # Login
        if username is None and "VISYM_COLLECTOR_EMAIL" in os.environ:
            username = os.environ["VISYM_COLLECTOR_EMAIL"]
        # Set user properties
        self._username = username
        self._password = password
        if password is not None:
            self.login(password)

        self.region_name = 'us-east-1'
        self.refresh()

    def refresh(self):
        if "VIPY_AWS_SESSION_TOKEN" in os.environ:
            self._set_S3_clients()
            self._set_lambda_clients()
            self._is_login = True
            self._cognito_username = os.environ["VIPY_AWS_COGNITO_USERNAME"]
        return self

    def login(self, password=None):
        """[summary]

        Args:
            username ([str]): username of pycollector.
            password ([str], optional): password for the user. Defaults to None.
        """

        username = self._username if self._username is not None else input("Collector email: ")
        assert is_email_address(username), 'Invalid collector email address "%s"' % username
        password = password if password is not None else getpass.getpass()

        try:
            # Set up API gateway request for login
            request_body = {"username": username, "password": password}
            self._aws_credentials = requests.post(
                GLOBALS["API_GATEWAY_HTTP"]["pycollector_login"],
                data=json.dumps(request_body),
                headers={"Content-type": "application/json", "Accept": "text/plain"},
            ).json()

            self._cognito_username = self._aws_credentials["cognito_username"]
            self._token_expiration_time = datetime.now() + timedelta(0, self._aws_credentials["token_expires_in_secs"])
            self.region_name = self._aws_credentials["region_name"]

            # Set up AWS services
            self._set_os_environ()
            self._set_parameter_store()
            self._set_S3_clients()
            self._set_lambda_clients()
            self._is_login = True

        except Exception as e:
            raise
            custom_error = "Failed to sign in due to exception: {0}".format(e)
            raise Exception(custom_error)

        return self

    def _set_parameter_store(self):
        """[summary]"""
        self._ssm_client = boto3.client(
            "ssm",
            aws_access_key_id=os.environ["VIPY_AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["VIPY_AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["VIPY_AWS_SESSION_TOKEN"],
            region_name=os.environ["VIPY_AWS_REGION"],
        )

    def get_ssm_param(self, param_name: str = None, WithDecryption: bool = False) -> str:
        """[summary]"""
        self._set_parameter_store()
        if self.is_token_expired():
            self.login()
        return self._ssm_client.get_parameter(Name=param_name, WithDecryption=WithDecryption).get("Parameter").get("Value")

    def _set_S3_clients(self):
        """[summary]"""
        assert "VIPY_AWS_SESSION_TOKEN" in os.environ
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["VIPY_AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["VIPY_AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["VIPY_AWS_SESSION_TOKEN"],
            region_name=os.environ["VIPY_AWS_REGION"],
        )
        self._s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=os.environ["VIPY_AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["VIPY_AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["VIPY_AWS_SESSION_TOKEN"],
            region_name=os.environ["VIPY_AWS_REGION"],
        )
        
    def _set_lambda_clients(self):
        """[summary]"""
        assert "VIPY_AWS_SESSION_TOKEN" in os.environ
        self._lambda_client = boto3.client(
            "lambda",
            aws_access_key_id=os.environ["VIPY_AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["VIPY_AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["VIPY_AWS_SESSION_TOKEN"],
            region_name=os.environ["VIPY_AWS_REGION"],
        )

    def _set_os_environ(self):
        """[summary]"""
        os.environ["VIPY_AWS_ACCESS_KEY_ID"] = self._aws_credentials["access_key_id"]
        os.environ["VIPY_AWS_SECRET_ACCESS_KEY"] = self._aws_credentials["secret_key"]
        os.environ["VIPY_AWS_SESSION_TOKEN"] = self._aws_credentials["session_token"]
        os.environ["VIPY_AWS_SESSION_TOKEN_EXPIRATION"] = str((datetime.now() + timedelta(0, self._aws_credentials["token_expires_in_secs"])).strftime("%Y-%m-%dT%H:%M:%S"))
        os.environ["VIPY_AWS_COGNITO_USERNAME"] = self._cognito_username
        os.environ["VIPY_AWS_REGION"] = self._aws_credentials["region_name"]

        
    def is_token_expired(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return "VIPY_AWS_SESSION_TOKEN_EXPIRATION" in os.environ and datetime.now() > pycollector.util.fromclockstamp(os.environ["VIPY_AWS_SESSION_TOKEN_EXPIRATION"])

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
        """"""
        return self._username

    @property
    def cognito_username(self):
        """"""
        return self._cognito_username

    @property
    def lambda_client(self):
        """"""
        return self._lambda_client
