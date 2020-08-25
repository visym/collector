import os
from email.utils import parseaddr
from datetime import datetime, timedelta, date
import pytz


def lowerif(s, b):
    return s.lower() if b else s

def isday(yyyymmdd):
    """Is the yyyymmdd formatted as 'YYYY-MM-DD' such as '2020-03-18'"""
    try:
        datetime.strptime(yyyymmdd, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def isdate(yyymmdd):
    """Alias for isday"""
    return isday(yymmdd)

def fromdate(yyyymmdd):
    return yyyymmdd_to_date(yyyymmdd)

def ismonday(yyyymmdd):
    if isday(yyyymmdd):
        return datetime.strptime(yyyymmdd, '%Y-%m-%d').weekday() == 0
    else:
        return False


def is_more_recent_than(yyyymmdd_1, yyyymmdd_2):
    assert isday(yyyymmdd_1) and isday(yyyymmdd_2), "Invalid input - must be 'YYYY-MM-DD'"
    return datetime.strptime(yyyymmdd_1, '%Y-%m-%d').date() >= datetime.strptime(yyyymmdd_2, '%Y-%m-%d').date()


def yyyymmdd_to_date(yyyymmdd):
    return datetime.strptime(yyyymmdd, '%Y-%m-%d').date()


def is_email_address(emailstr, strict=False):
    """Quick and dirty, will throw an error on blank characters if strict=True"""
    return '@' in emailstr and '.' in emailstr and "'" not in emailstr and '"' not in emailstr and (strict is False or " " not in emailstr)


def allmondays(year=datetime.today().year):
    assert isinstance(year, int), "year must be int(YYYY), such as int(2020)"
    d = date(year, 1, 1)                # January 1st
    d += timedelta(days = 7-d.weekday())  # First monday
    
    mondays = []
    while d.year == year:
        mondays.append(str(d))
        d += timedelta(days = 7)
    return mondays


def timestamp_YYYYMMDD_HHMMSS():
    """Datetime stamp in eastern timezone with second resolution"""
    return datetime.now().astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %H:%M:%S")    

def istimestamp_YYYYMMDD_HHMMSS(t):
    try:
        fromtimestamp_YYYYMMDD_HHMMSS(t)
        return True
    except:
        return False

def fromtimestamp_YYYYMMDD_HHMMSS(ts):
    """Assumed eastern timezone"""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    
def timestamp_for_gsheets():
    """Datetime stamp in eastern timezone suitable to write so that gsheets data validation of date does not throw a warning"""        
    return timestamp_YYYYMMDD_HHMMSS()

def timestamp():
    """Datetime stamp in eastern timezone with microsecond resolution"""    
    return datetime.now().astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%dT%H:%M:%S.%f%z")

def istimestamp(t):
    try:
        fromtimestamp(t)
        return True
    except:
        return False

def fromtimestamp(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")

    
def clockstamp():
    """Datetime stamp in eastern timezone with second resolution"""        
    return datetime.now().astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%dT%H:%M:%S%z")    

def datestamp():
    return datetime.now().strftime("%Y-%b-%d")    

def today():
    return datetime.now().date()

def fromclockstamp(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

def allmondays_since(yyyymmdd):
    assert isday(yyyymmdd)
    since = datetime.strptime(yyyymmdd, '%Y-%m-%d')
    lastmonday = since - timedelta(days=since.weekday())
    mondays = allmondays(since.year)
    return [m for m in mondays if datetime.strptime(m, '%Y-%m-%d') >= lastmonday and datetime.strptime(m, '%Y-%m-%d') <= datetime.now()]
    
def lastmonday(yyyymmdd=None):
    since = datetime.strptime(yyyymmdd, '%Y-%m-%d') if yyyymmdd is not None else datetime.now()
    return str((since - timedelta(days=since.weekday())).date())

def lastweek_monday(yyyymmdd=None):
    since = datetime.strptime(yyyymmdd, '%Y-%m-%d') if yyyymmdd is not None else datetime.today()
    return str((since + timedelta(days=-since.weekday(), weeks=-1)).date())

def nextsunday(yyyymmdd=None):
    return str((yyyymmdd_to_date(lastmonday(yyyymmdd)) + timedelta(days=6)))


def nextday(yyyymmdd=None):
    return str(yyyymmdd_to_date(yyyymmdd) + timedelta(days=1))


def get_reviewer_id(reviewer_id=None):
    if reviewer_id is None and 'VISYM_COLLECTOR_REVIEWER_ID' in os.environ:
        reviewer_id = os.environ['VISYM_COLLECTOR_REVIEWER_ID']
    elif reviewer_id is None:
        raise ValueError("Please set up environment variable: VISYM_COLLECTOR_REVIEWER_ID with a valid email address")
    assert is_email_address(reviewer_id), "Invalid email address '%s'" % reviewer_id
    return reviewer_id

    


    
