import os
import builtins
import logging as python_logging
import warnings
import vipy.globals
from vipy.util import try_import


# Global mutable dictionary
GLOBALS = {'VERBOSE': True,                # If False, will silence everything, equivalent to calling pycollector.globals.silent()
           'LOGGING':False,                # If True, use python logging (handler provided by end-user) intead of print 
           'LOGGER':None,                  # The global logger used by pycollector.globals.print() and pycollector.globals.warn() if LOGGING=True
           'BACKEND_VERSION':'v2'}         


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


def backend(env=None, version=None, flush=False):
    assert vipy.version.is_at_least('1.8.27')
    try_import('pycollector.admin', message="Not authorized")
    import pycollector.admin.globals
    return pycollector.admin.globals.backend(env, version, flush)

    
def verbose(b=None):
    if b is not None:
        GLOBALS['VERBOSE'] = b
    return GLOBALS['VERBOSE']


def silent():
    GLOBALS['VERBOSE'] = False    

    
def isapi(version):
    return GLOBALS['BACKEND_VERSION'] == version

