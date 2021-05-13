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

    def get_project(
        self,
        program_id="Practice",
        project_id=None,
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
            project_id (str, optional): [description]. Defaults to 'MEVA'.
        """

        return Project(
            program_id=program_id,
            project_id=project_id,
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

    def get_recent_videos(self, program_id="MEVA", n=1):
        """[summary]

        Args:
            project_id (str, optional): [description]. Defaults to 'MEVA'.
        """

        return Project(program_id=program_id, last=n).last(n)

    def new_collection(
        self,
        name,
        # program_name,
        project_name,
        collection_description,
        activities,
        activity_short_names,
        objects,
        actor,
        consent_overlay_text='Please select the record button, say "I consent to this video collection”',
    ):

        # Invoke Lambda function
        request = {
            "collection_name": name,
            "activities": activities,  # comma separated string, no spaces
            "project_name": project_name,
            # "program_name": program_name,
            "objects": objects,
            "actor": actor,
            "collection_description": collection_description,
            "activity_short_names": activity_short_names,  # comma separated string, no spaces
            "creator_cognito_username": self.cognito_username,
            "consent_overlay_text": consent_overlay_text,
        }

        # Invoke Lambda function
        try:
            FunctionName = self.get_ssm_param(GLOBALS["LAMBDA"]["create_collection"])
            response = self._lambda_client.invoke(
                FunctionName=FunctionName,
                InvocationType="RequestResponse",
                LogType="Tail",
                Payload=json.dumps(request),
            )

            # # Get the serialized message
            # dict_str = response["Payload"].read().decode("UTF-8")
            # if dict_str == "null":
            #     raise ValueError("Invalid lambda function response")
            # data_dict = ast.literal_eval(dict_str)

            # print("data_dict", data_dict)
            print("Collection '{}' added successfully".format(name))

        except Exception as e:
            custom_error = "\nException : failed to invoke lambda function to create collection.\n"
            custom_error += "Error : " + str(e) + "\n"
            raise Exception(custom_error)

    def list_collections(self):

        # Invoke Lambda function
        request = {"collectorid": self.cognito_username}

        # Invoke Lambda function
        try:
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
            data_df = pd.read_json(serialized_collections_data_dict)
            return data_df

        except Exception as e:
            custom_error = "\nException : failed to invoke lambda function to list collections.\n"
            custom_error += "Error : " + str(e) + "\n"
            raise Exception(custom_error)

    def delete_collection(self, collectionid):

        # Invoke Lambda function
        request = {"collectorid": self.cognito_username, "collectionid": collectionid}

        # Invoke Lambda function
        try:
            FunctionName = self.get_ssm_param(GLOBALS["LAMBDA"]["delete_collection"])
            response = self._lambda_client.invoke(
                FunctionName=FunctionName,
                InvocationType="RequestResponse",
                LogType="Tail",
                Payload=json.dumps(request),
            )

            # Get the serialized message
            dict_str = response["Payload"].read().decode("UTF-8")
            if dict_str == "null":
                raise ValueError("Invalid lambda function response")
            data_dict = ast.literal_eval(dict_str)

            print(data_dict["body"]["message"])

        except Exception as e:
            custom_error = "\nException : failed to invoke lambda function to delete collection.\n"
            custom_error += "Error : " + str(e) + "\n"
            raise Exception(custom_error)
