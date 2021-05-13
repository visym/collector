import os
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json

# internal testing modules
# from pycollector.user import User
from pycollector.project import Project

import pycollector


def _test_get_project_visym(org='visym', env='prod'):
    """test get project by user"""

    # Set to target ENV
    if org != 'visym' or env != 'prod':
        # Non-default backend requires admin tools
        from pycollector.admin.globals import backend    
        pycollector.admin.globals.backend(env="prod", org="visym")

    # testing variiables
    username = os.environ["VISYM_COLLECTOR_PYTEST_EMAIL"]  # github secrets
    password = os.environ["VISYM_COLLECTOR_PYTEST_PASSWORD"]  # github secrets

    #username = "heng2j@gmail.com"  # "zhongheng.li@stresearch.com"
    #password = None  # sanitized

    # testing objects and functions
    new_project_client = Project(program_id="Practice", alltime=True, username=username, password=password)

    # Fetching Videos
    videos = new_project_client.videos()
    assert len(videos) >= 1

    # quickshow video
    videos[0].show()
