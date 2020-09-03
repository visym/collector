import os
import builtins
import logging as python_logging
import warnings
import vipy.globals


# Global mutable dictionary
GLOBALS = {'VERBOSE': True,                # If False, will silence everything, equivalent to calling vipy.globals.silent()
           'LOGGING':False,                # If True, use python logging (handler provided by end-user) intead of print 
           'LOGGER':None,                  # The global logger used by vipy.globals.print() and vipy.globals.warn() if LOGGING=True
           'BACKEND_VERSION': 'v2',        # For visym admins only - Do not change
           'BACKEND_ENVIRONMENT': 'prod',  # For visym admins only - Do not change
           'BACKEND':None}   


def logging(enable=None, format=None):
    """Single entry point for enabling/disabling logging vs. printing
       
       All vipy functions overload "from vipy.globals import print" for simplified readability of code.
       This global function redirects print or warn to using the standard logging module.
       If format is provided, this will create a basicConfig handler, but this should be configured by the end-user.    
    """
    if enable is not None:
        assert isinstance(enable, bool)
        GLOBALS['LOGGING'] = enable
        if format is not None:
            python_logging.basicConfig(level=python_logging.INFO, format=format)
        GLOBALS['LOGGER'] = python_logging.getLogger('pycollector')
        GLOBALS['LOGGER'].propagate = True if enable else False
        vipy.globals.logging(True)
        
    return GLOBALS['LOGGING']


def warn(s):
    if GLOBALS['VERBOSE']:
        warnings.warn(s) if (not GLOBALS['LOGGING'] or GLOBALS['LOGGER'] is None) else GLOBALS['LOGGER'].warn(s)

        
def print(s, end='\n'):
    """Main entry point for all print statements in the pycollector package. All pycollector code calls this to print helpful messages.
      
       -Printing can be disabled by calling vipy.globals.silent()
       -Printing can be redirected to logging by calling vipy.globals.logging(True)
       -All print() statements in vipy.* are overloaded to call vipy.globals.print() so that it can be redirected to logging

    """
    if GLOBALS['VERBOSE']:
        builtins.print(s, end=end) if (not GLOBALS['LOGGING'] or GLOBALS['LOGGER'] is None) else GLOBALS['LOGGER'].info(s)

        
def verbose(b=None):
    if b is not None:
        GLOBALS['VERBOSE'] = b
    return GLOBALS['VERBOSE']


def silent():
    GLOBALS['VERBOSE'] = False    

    
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
