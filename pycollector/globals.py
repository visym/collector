import os


GLOBALS = {'BACKEND_VERSION': 'v2',
           'BACKEND_ENVIRONMENT': 'prod',           
           'BACKEND':None,
           'VERBOSE':True}


def verbose(b=None):
    if b is not None:
        GLOBALS['VERBOSE'] = b
    return GLOBALS['VERBOSE']
    
    
def backend(env=None, version=None, flush=False):
    if version is not None:
        assert version in ['v1', 'v2']
        GLOBALS['BACKEND_VERSION'] = version
        GLOBALS['BACKEND'] = None

    if env is not None:
        assert env in ['prod', 'dev', 'test']
        GLOBALS['BACKEND_ENVIRONMENT'] = env
        GLOBALS['BACKEND'] = None

    if flush:
        GLOBALS['BACKEND'] = None        
        
    if GLOBALS['BACKEND'] is None:
        import pycollector.backend  # avoid circular import
        if GLOBALS['BACKEND_VERSION'] == 'v1' and GLOBALS['BACKEND_ENVIRONMENT'] == 'prod':
            GLOBALS['BACKEND'] = pycollector.backend.Prod_v1()
        elif GLOBALS['BACKEND_VERSION'] == 'v2' and GLOBALS['BACKEND_ENVIRONMENT'] == 'prod':
            GLOBALS['BACKEND'] = pycollector.backend.Prod()
        elif GLOBALS['BACKEND_VERSION'] == 'v2' and GLOBALS['BACKEND_ENVIRONMENT'] == 'test':
            GLOBALS['BACKEND'] = pycollector.backend.Test()
        else:
            raise ValueError('Invalid env=%s version=%s, must be env=[prod, test] and version=[v1,v2]' % (env, version))
            
    return GLOBALS['BACKEND']


def api(version):
    return backend(version=version)


def isapi(version):
    return GLOBALS['BACKEND_VERSION'] == version

def isprod():
    return GLOBALS['BACKEND_ENVIRONMENT'] == 'prod'
