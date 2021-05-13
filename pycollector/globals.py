import os
import builtins
import logging as python_logging
import warnings
import vipy.globals


# Global mutable dictionary
GLOBALS = {
    "VERBOSE": True,  # If False, will silence everything, equivalent to calling pycollector.globals.silent()
    "LOGGING": False,  # If True, use python logging (handler provided by end-user) intead of print
    "LOGGER": None,  # The global logger used by pycollector.globals.print() and pycollector.globals.warn() if LOGGING=True
    "API_GATEWAY_HTTP": {"pycollector_login": "https://7ezps88yk3.execute-api.us-east-1.amazonaws.com/pycollector_login"},
    "LAMBDA": {
        "get_project": "pycollector_lambda_get_project",
        "list_collections": "pycollector_list_collections",
        "create_collection": "pycollector_create_new_collection",
        "delete_collection": "pycollector_delete_collection",
    },
}


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
        GLOBALS["LOGGING"] = enable
        if format is not None:
            python_logging.basicConfig(level=python_logging.INFO, format=format)
        GLOBALS["LOGGER"] = python_logging.getLogger("pycollector")
        GLOBALS["LOGGER"].propagate = True if enable else False
        vipy.globals.logging(True)

    return GLOBALS["LOGGING"]


def warn(s):
    if GLOBALS["VERBOSE"]:
        warnings.warn(s) if (not GLOBALS["LOGGING"] or GLOBALS["LOGGER"] is None) else GLOBALS["LOGGER"].warn(s)


def print(s, end="\n"):
    """Main entry point for all print statements in the pycollector package. All pycollector code calls this to print helpful messages.

    -All print() statements in pycollector are overloaded to call pycollector.globals.print() so that it can be redirected to logging as needed
    -Printing can be disabled by calling pycollector.globals.silent()
    -Printing can be redirected to standard python logging by calling pycollector.globals.logging(True)


    """
    if GLOBALS["VERBOSE"]:
        builtins.print(s, end=end) if (not GLOBALS["LOGGING"] or GLOBALS["LOGGER"] is None) else GLOBALS["LOGGER"].info(s)


def verbose(b=None):
    if b is not None:
        GLOBALS["VERBOSE"] = b
    return GLOBALS["VERBOSE"]


def silent():
    GLOBALS["VERBOSE"] = False
