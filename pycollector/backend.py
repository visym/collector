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
        if (not self.is_authenticated()) or self.is_token_expired():
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
    

    def new_collection(self, name, organization_name, program_name, project_name, description, activities, activity_short_names, objects, creator_cognito_username):

        # Invoke Lambda function
        request = {'name': name, 
                    'organization_name': organization_name, 
                    'program_name': program_name,
                    'project_name': project_name, 
                    'description': description,
                    'activities': activities, 
                    'activity_short_names': activity_short_names,
                    'objects': objects,
                    'creator_cognito_username': creator_cognito_username,
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
