import os
import builtins
import logging as python_logging
import warnings
import vipy.globals


# Global mutable dictionary
GLOBALS = {'VERBOSE': True,                # If False, will silence everything, equivalent to calling pycollector.globals.silent()
           'LOGGING':False,                # If True, use python logging (handler provided by end-user) intead of print 
           'LOGGER':None,                  # The global logger used by pycollector.globals.print() and pycollector.globals.warn() if LOGGING=True
           'LAMBDA':{'get_project':'arn:aws:lambda:us-east-1:806596299222:function:pycollector_get_project'},
           'COGNITO':{'app_client_id': '7tge9bc9e3iv9r2i644dakr7qp', # '6k20qruljfs0v7n5tmt1pk0u1q',
                      'identity_pool_id': 'us-east-1:efb75fcb-9009-4b4a-959c-9eec61e19359', #'us-east-1:c7bbbc40-37d3-4ad8-8afd-492c095729bb',
                      'provider_name': 'cognito-idp.us-east-1.amazonaws.com/us-east-1_LdCiaH9IZ', #'cognito-idp.us-east-1.amazonaws.com/us-east-1_sFpJQRLiY',
                      'region_name':'us-east-1' if 'AWS_DEFAULT_REGION' not in os.environ else os.environ['AWS_DEFAULT_REGION']}}


def cachedir(newdir):
    """Set the location to save videos and JSON files when downloaded.  This will default to the system temp directory if not set.
    
       -This can be set by default by creating the environment variable VIPY_CACHE='/path/to/newdir'
    """
    vipy.globals.cache(newdir)

    
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

       -All print() statements in pycollector are overloaded to call pycollector.globals.print() so that it can be redirected to logging as needed      
       -Printing can be disabled by calling pycollector.globals.silent()
       -Printing can be redirected to standard python logging by calling pycollector.globals.logging(True)


    """
    if GLOBALS['VERBOSE']:
        builtins.print(s, end=end) if (not GLOBALS['LOGGING'] or GLOBALS['LOGGER'] is None) else GLOBALS['LOGGER'].info(s)


def verbose(b=None):
    if b is not None:
        GLOBALS['VERBOSE'] = b
    return GLOBALS['VERBOSE']


def silent():
    GLOBALS['VERBOSE'] = False    


