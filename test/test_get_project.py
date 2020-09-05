# External modules
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json

# internal testing modules
from pycollector.user import User
from pycollector.project import Project


def test_get_project():
    """ test get project by user
    """
    # testing variiables
    username = 'zhongheng.li@stresearch.com'
    password = '0STRBoston&0'

    # testing objects and functions 
    user = User(username=username, password=password)

    project_client = Project(program_id="MEVA", alltime=True, pycollector=user)


    print("print collector ids")
    print(new_project_client.collectorID())

    print("print collectoremail")
    print(new_project_client.collectoremail())


    # Fetching Videos
    videos = new_project_client.videos()

    # Fetching instances
    instances = new_project_client.instances()
    print(instances)



