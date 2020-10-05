import os
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json
from pycollector.backend import API 

# testing variables
_USER_NAME = os.environ['VISYM_COLLECTOR_PYTEST_EMAIL']
_PASSWORD = os.environ['VISYM_COLLECTOR_PYTEST_PASSWORD']
_PROGRAM_ID = 'MEVA'


@pytest.mark.parametrize("username, password", [(_USER_NAME, _PASSWORD)])
def test_get_user_video(username,password):
    """test get user video
    """
    # testing objects and functions 
    api = API(username=username, password=password)
    assert api.username == username
    assert api.is_token_expired() == False


@pytest.mark.parametrize("username, password, program_id", [(_USER_NAME, _PASSWORD, _PROGRAM_ID)])
def test_get_user_videos(username,password, program_id):
    """test get user video
    """
    
    # testing objects and functions 
    api = API(username=username, password=password)
    project =  api.get_project(program_id=program_id)
    assert len(project) > 0
    
    # show the df of project
    print(project.df)

    # fetch video
    videos = project.videos()
    assert len(videos) > 0
        
    # Download video 
    videos[0].download() 



@pytest.mark.parametrize("username, password, program_id", [(_USER_NAME, _PASSWORD, _PROGRAM_ID)])
def test_new_collections(username,password, program_id):
    """test add  new collection
    """
    
    # testing objects and functions 
    api = API(username=username, password=password)

    # test to create new collection
    name = 'Thank you'
    organization_name = 'ARSl'
    program_name = 'ARSL'
    project_name = 'ARSL_American Sign Language - Purchasing'
    description = 'ASL for asking some if the person needs help. \n\n The sign for \"help\" is made by forming a loose-thumb-A hand (or even an \"S\" hand) and lifting it with your other hand. Some people will tell you the \"A\" hand should be your right hand. Others will tell you it should be your left hand. The reality of the matter is if you look this sign up in a half-dozen different sources you are going to see it done several different ways. \n\n  HELP: You / Need Help / '
    activities = 'thank you'
    activity_short_names = 'Thank you'
    objects = 'Person'
    cognito_username =  'test_cognito_username'

    api.new_collection(name, organization_name, program_name, project_name, description, activities, activity_short_names, objects, cognito_username)

