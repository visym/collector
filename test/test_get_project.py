import os
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
    #pycollector.globals.backend('test')

    # testing variiables
    username = os.environ['VISYM_COLLECTOR_PYTEST_EMAIL']  # github secrets
    password = os.environ['VISYM_COLLECTOR_PYTEST_PASSWORD']  # github secrets

    # testing objects and functions 
    new_project_client = Project(program_id="MEVA", alltime=True, username=username, password=password)

    # Fetching Videos
    videos = new_project_client.videos()
    assert len(videos) >= 1

    # # quickshow video
    # videos[0].quickshow()

