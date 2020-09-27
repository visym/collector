from pycollector.user import User
from pycollector.project import Project



class API(User):
    """ class for pycollector API

    Args:
        object ([type]): [description]
    """


    def __init__(self, username=None, password=None, test=False):
        """
    
        Args:
            username ([type], optional): [description]. Defaults to None.
            password ([type], optional): [description]. Defaults to None.
        """
        super().__init__(username=username,password=password,  test=False)


    def get_project(self, program_id='MEVA'): 
        """[summary]

        Args:
            project_id (str, optional): [description]. Defaults to 'MEVA'.
        """

        return Project(program_id=program_id, alltime=True, pycollector=self)


    def new_collection(self, name, organization_name, program_name,project_name, description, activities, activity_short_names, objects ):

        # Invoke Lambda function
        request = {'identity_id': identity_id, 'cognito_username': cognito_username, 'email': email}


        # Invoke Lambda function
        try:
            response = self._lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:806596299222:function:CollectorTestPostAuthentication',
            InvocationType= 'RequestResponse',
            LogType='Tail',
            Payload=json.dumps(request),
        )
        except Exception as e:
            custom_error = '\nException : failed to invoke jobs.\n'
            custom_error += 'Error : ' + str(e) + '\n'
            raise Exception(custom_error)