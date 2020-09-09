# External modules
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json

# internal testing modules
from pycollector.user import User
from pycollector.project import Project
import pycollector.globals

def test_get_project():
    """ test get project by user
    """

    # Set to target ENV
    pycollector.globals.backend('test')

    # testing variiables
    username = 'zhongheng.li@stresearch.com'
    password = '0STRBoston&0'

    # testing objects and functions 
    user = User(username=username, password=password)
    new_project_client = Project(program_id="MEVA", alltime=True, pycollector=user)


    print("print collector ids")
    print(new_project_client.collectorID())

    print("print collectoremail")
    print(new_project_client.collectoremail())

    print(new_project_client.df)


    # Fetching Videos
    videos = new_project_client.videos()

    # Fetching instances
    instances = new_project_client.instances()
    print(instances)



