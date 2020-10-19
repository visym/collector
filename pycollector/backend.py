import json
from pycollector.user import User
from pycollector.project import Project



class API(User):
    """ class for pycollector API

    Args:
        object ([type]): [description]
    """


    def __init__(self, username=None, password=None):
        """
    
        Args:
            username ([type], optional): [description]. Defaults to None.
            password ([type], optional): [description]. Defaults to None.
        """
        super().__init__(username=username,password=password)
        if not self.is_authenticated():
            self.login()
        

    def get_project(self, program_id='MEVA'): 
        """[summary]

        Args:
            project_id (str, optional): [description]. Defaults to 'MEVA'.
        """

        return Project(program_id=program_id, alltime=True)

    
    def get_recent_videos(self, program_id='MEVA', n=1): 
        """[summary]

        Args:
            project_id (str, optional): [description]. Defaults to 'MEVA'.
        """

        return Project(program_id=program_id, last=n).last(n)
    

    def new_collection(self, name, program_name, project_name, collection_description, activities, activity_short_names, objects, actor, consent_overlay_text='Please select the record button, say "I consent to this video collection”'):

        # Invoke Lambda function
        request = {'collection_name':name,
                    'activities': activities,  # comma separated string, no spaces
                    'project_name':project_name,
                    'program_name':program_name,
                    'objects': objects,
                    'actor' : actor,
                    'collection_description': collection_description,
                    'activity_short_names': activity_short_names, # comma separated string, no spaces
                    'creator_cognito_username' : self.cognito_username,
                    'consent_overlay_text': consent_overlay_text,
                    }

        # Invoke Lambda function
        try:
            response = self._lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:806596299222:function:pycollector_create_new_collection',
            InvocationType= 'RequestResponse',
            LogType='Tail',
            Payload=json.dumps(request),
        )
        except Exception as e:
            custom_error = '\nException : failed to invoke jobs.\n'
            custom_error += 'Error : ' + str(e) + '\n'
            raise Exception(custom_error)

    def delete_collection(self, collectionid):

        # Invoke Lambda function
        request = {'collectorid': self.cognito_username,
                    'collectionid': collectionid
                    }

        # Invoke Lambda function
        try:
            response = self._lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:806596299222:function:pycollector_delete_collection',
            InvocationType= 'RequestResponse',
            LogType='Tail',
            Payload=json.dumps(request),
        )
        except Exception as e:
            custom_error = '\nException : failed to invoke jobs.\n'
            custom_error += 'Error : ' + str(e) + '\n'
            raise Exception(custom_error)




