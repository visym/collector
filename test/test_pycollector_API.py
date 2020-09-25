
# External modules
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json


from pycollector.backend import API 

# testing variables
_USER_NAME = 'zhongheng.li@stresearch.com'
_PASSWORD = '0STRBoston&0'
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

    print(project.df)

    videos = project.videos()
