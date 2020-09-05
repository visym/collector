
# External modules
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import json

# internal testing modules
from pycollector.user import User

def test_user_sign():
    """ test user sign in 
    """
    # testing variiables
    username = 'zhongheng.li@stresearch.com'
    password = '0STRBoston&0'

    # testing objects and functions 
    user = User(username=username, password=password)
    assert user.username == 'zhongheng.li@stresearch.com'
    assert user.is_token_expired() == False







