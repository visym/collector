MAJOR = 0
MINOR = 2
RELEASE = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, RELEASE)

GLOBALS = {'API_VERSION': 'v2'}


def api(version=None):
    if version is not None:
        assert version in ['v1', 'v2', 'v2-android']
        GLOBALS['API_VERSION'] = version
    return GLOBALS['API_VERSION'] 
    

def isapi(version):
    return version in GLOBALS['API_VERSION']


