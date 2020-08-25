import os
import pycollector.backend


GLOBALS = {'BACKEND_VERSION': 'v2',
           'BACKEND':{}}

def backend(version=None):
    if version is not None:
        assert version in ['v1', 'v2', 'dev', 'test']
        GLOBALS['BACKEND_VERSION'] = version
        
    if GLOBALS['BACKEND_VERSION'] == 'v1' and version not in GLOBALS['BACKEND']:
        GLOBALS['BACKEND'][version] = pycollector.backend.Prod_v1()
    elif GLOBALS['BACKEND_VERSION'] == 'v2' and version not in GLOBALS['BACKEND']:
        GLOBALS['BACKEND'][version] = pycollector.backend.Prod()
    return GLOBALS['BACKEND'][version]


def api(version):
    return backend(version)


def isapi(version):
    return version == GLOBALS['BACKEND_VERSION']
