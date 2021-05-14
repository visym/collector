import json
import pandas as pd
import ast
from pycollector.user import User
from pycollector.project import Project
from pycollector.globals import print, GLOBALS


class API(User):
    """class for pycollector API

    Args:
        object ([type]): [description]
    """

    def __init__(self, username=None, password=None):
        """

        Args:
            username ([type], optional): [description]. Defaults to None.
            password ([type], optional): [description]. Defaults to None.
        """
        super().__init__(username=username, password=password)
        if not self.is_authenticated():
            self.login()

    def get_project(self,
                    project=None,
                    alltime=True,
                    weeksago=None,
                    monthsago=None,
                    daysago=None,
                    since=None,
                    before=None,
                    last=None,
                    retry=2,
    ):
        """[summary]

        Args:
            -
        """

        return Project(
            project=project,
            alltime=alltime,
            weeksago=weeksago,
            monthsago=monthsago,
            daysago=daysago,
            since=since,
            before=before,
            last=last,
            retry=retry,
            username=self._username,
            password=self._password,
        )

    def videos(self, n):
        """[summary]

        Args:
            -
        """

        return Project(last=n).last(n)

    def lastvideo(self):
        return self.videos(n=1)
    
    def new_collection(
            self,
            name,
            # program_name,
            project_name,
            collection_description,
            activities,
            activity_short_names,
            actor,
            consent_overlay_text='Please select the record button, say "I consent to this video collection”',
            objects=None,            
    ):

        assert isinstance(name, str)
        assert isinstance(project_name, str)
        assert isinstance(collection_description, str)
        assert isinstance(activities, str) or (isinstance(activities, tuple) and all([isinstance(a, str) for a in activities]))
        assert isinstance(activity_short_names, str) or (isinstance(activity_short_names, tuple) and all([isinstance(a, str) for a in activity_short_names]))
        assert (isinstance(activities, str) and isinstance(activity_short_names, str)) or len(activities) == len(activity_short_names)
        assert isinstance(actor, str)
        assert objects is None or isinstance(objects, str) or (isinstance(objects, tuple) and all([isinstance(o, str) for o in objects]))
                                               
        activities_csv = ','.join(activities) if isinstance(activities, tuple) else activities
        activity_short_names_csv = ','.join(activity_short_names) if isinstance(activity_short_names, tuple) else activity_short_names
        objects = () if objects is None else objects
        objects_csv = ','.join(objects) if isinstance(objects, tuple) else objects
        
        request = {
            "collection_name": name,
            "activities": activities_csv,  # comma separated string, no spaces
            "project_name": project_name,
            # "program_name": program_name,
            "objects": objects_csv,  # comma separated string, no spaces
            "actor": actor,
            "collection_description": collection_description,
            "activity_short_names": activity_short_names_csv,  # comma separated string, no spaces
            "creator_cognito_username": self.cognito_username,
            "consent_overlay_text": consent_overlay_text,  # currently disabled
        }

        # Invoke Lambda function
        FunctionName = self.get_ssm_param(GLOBALS["LAMBDA"]["create_collection"])
        response = self._lambda_client.invoke(
            FunctionName=FunctionName,
            InvocationType="RequestResponse",
            LogType="Tail",
            Payload=json.dumps(request),
        )

        d = json.loads(response['Payload'].read().decode('UTF-8'))
        if 'statusCode' not in d or d['statusCode'] != 200:
            raise ValueError(str(d))
        else:
            return d

    def list_collections(self):

        # Invoke Lambda function
        request = {"collectorid": self.cognito_username}

        # Invoke Lambda function
        FunctionName = self.get_ssm_param(GLOBALS["LAMBDA"]["list_collections"])
        response = self._lambda_client.invoke(
            FunctionName=FunctionName,
            InvocationType="RequestResponse",
            LogType="Tail",
            Payload=json.dumps(request),
        )
        
        # Get the serialized dataframe
        dict_str = response["Payload"].read().decode("UTF-8")
        if dict_str == "null":
            raise ValueError("Invalid lambda function response")
        data_dict = ast.literal_eval(dict_str)
        
        serialized_collections_data_dict = data_dict["body"]["collections"]

        # Return list of dictionaries
        h = ['created_date', 'name', 'collection_id', 'project_name']
        d = {k:v for (k,v) in json.loads(serialized_collections_data_dict).items() if k in h}
        n = len(d['name']) if 'name' in d else 0
        return [{k:v[str(j)] for (k,v) in d.items()} for j in range(0,n)]


    def delete_collection(self, collectionid):

        assert isinstance(collectionid, str)
        
        # Invoke Lambda function
        request = {"collectorid": self.cognito_username, "collectionid": collectionid}

        # Invoke Lambda function
        FunctionName = self.get_ssm_param(GLOBALS["LAMBDA"]["delete_collection"])
        response = self._lambda_client.invoke(
            FunctionName=FunctionName,
            InvocationType="RequestResponse",
            LogType="Tail",
            Payload=json.dumps(request),
        )
        
        # Get the serialized message
        d = json.loads(response['Payload'].read().decode('UTF-8'))
        if 'statusCode' not in d or d['statusCode'] != 200:
            raise ValueError(str(d))
        else:
            return d
            
