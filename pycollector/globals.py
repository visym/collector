import os
from pycollector.backend import Backend


GLOBALS = {'BACKEND_VERSION': 'v2',
           'BACKEND_ENVIRONMENT': 'prod',           
           'BACKEND':None,
           'VERBOSE':True}


class TestBackend(Backend):
    def __init__(self):
        super(TestBackend, self).__init__()        
        self._s3_bucket = 'diva-prod-data-lake232615-visymtest'
        self._ddb_video = self._dynamodb_resource.Table('strVideos-uxt26i4hcjb3zg4zth4uaed4cy-visymtest')
        self._ddb_instance = self._dynamodb_resource.Table("strInstances-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_rating = self._dynamodb_resource.Table("strRating-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_program = self._dynamodb_resource.Table("strProgram-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_collection = self._dynamodb_resource.Table("strCollections-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_project = self._dynamodb_resource.Table("strProjects-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")    
        self._ddb_activity = self._dynamodb_resource.Table("strActivities-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_subject = self._dynamodb_resource.Table("strSubject-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_collector = self._dynamodb_resource.Table("strCollector-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        self._ddb_collection_assignment = self._dynamodb_resource.Table("strCollectionsAssignment-uxt26i4hcjb3zg4zth4uaed4cy-visymtest")
        

class DevBackend(object):
    def __init__(self):
        pass

    
class Prod_v1_Backend(Backend):
    def __init__(self):
        super(Prod_v1_Backend, self).__init__()
        self._ddb_instance = self._dynamodb_resource.Table("co_Instances_dev")
        self._ddb_rating = self._dynamodb_resource.Table("co_Rating")
        self._ddb_program = self._dynamodb_resource.Table("co_Programs")
        self._ddb_video = self._dynamodb_resource.Table('co_Videos')    
        
    
class ProdBackend(Backend):
    def __init__(self):
        super(ProdBackend, self).__init__()        
        self._s3_bucket = 'diva-prod-data-lake174516-visym'
        self._ddb_video = self._dynamodb_resource.Table('strVideos-hirn6lrwxfcrvl65xnxdejvftm-visym')
        self._ddb_instance = self._dynamodb_resource.Table("strInstances-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_rating = self._dynamodb_resource.Table("strRating-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_program = self._dynamodb_resource.Table("strProgram-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_collection = self._dynamodb_resource.Table("strCollections-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_project = self._dynamodb_resource.Table("strProjects-hirn6lrwxfcrvl65xnxdejvftm-visym")    
        self._ddb_activity = self._dynamodb_resource.Table("strActivities-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_subject = self._dynamodb_resource.Table("strSubject-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_collector = self._dynamodb_resource.Table("strCollector-hirn6lrwxfcrvl65xnxdejvftm-visym")
        self._ddb_collection_assignment = self._dynamodb_resource.Table("strCollectionsAssignment-hirn6lrwxfcrvl65xnxdejvftm-visym")

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
        if GLOBALS['BACKEND_VERSION'] == 'v1' and GLOBALS['BACKEND_ENVIRONMENT'] == 'prod':
            GLOBALS['BACKEND'] = Prod_v1_Backend()
        elif GLOBALS['BACKEND_VERSION'] == 'v2' and GLOBALS['BACKEND_ENVIRONMENT'] == 'prod':
            GLOBALS['BACKEND'] = ProdBackend()
        elif GLOBALS['BACKEND_VERSION'] == 'v2' and GLOBALS['BACKEND_ENVIRONMENT'] == 'test':
            GLOBALS['BACKEND'] = TestBackend()
        else:
            raise ValueError('Invalid env=%s version=%s, must be env=[prod, test] and version=[v1,v2]' % (env, version))
            
    return GLOBALS['BACKEND']


def api(version):
    return backend(version=version)


def isapi(version):
    return GLOBALS['BACKEND_VERSION'] == version

def isprod():
    return GLOBALS['BACKEND_ENVIRONMENT'] == 'prod'
