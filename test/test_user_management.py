import os
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json

# internal testing modules
from pycollector.user import User


def test_user_login():
    
    # testing variiables
    username = os.environ['VISYM_COLLECTOR_PYTEST_EMAIL']  # github secrets
    password = os.environ['VISYM_COLLECTOR_PYTEST_PASSWORD']  # github secrets

    # testing objects and functions 
    user = User(username=username, password=password)
    assert user.is_token_expired() == False

    print('[test_user_login]: PASSED')





