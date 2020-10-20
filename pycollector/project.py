import os
import vipy
import warnings
import numpy as np
import pandas as pd
import json
import ast

from pycollector.globals import print, GLOBALS
from pycollector.video import Video
from pycollector.user import User


class Project(User):
    """collector.project.Project class

       Projects() are sets of CollectionInstances() and Instances() in a program.
    """
    
    def __init__(self,
                 program_id='MEVA',
                 project_id=None,
                 weeksago=None,
                 monthsago=None,
                 daysago=None,
                 since=None,
                 before=None,
                 alltime=False,
                 last=None,
                 retry=2,
                 username=None,
                 password=None
    ):
        super().__init__(username=username, password=password)
        
        if not self.refresh().is_authenticated():
            self.login() 
        
        self._projects = None
        self._programid = program_id

        # Get data from backend lambda function
        # Invoke Lambda function
        request = {'program_id': program_id, 'project_id': project_id, 'weeksago': weeksago, 'monthsago': monthsago, 'daysago': daysago, 'since': since, 'alltime': alltime, 'Video_IDs': None, 'before': before, 'week': None, 'pycollector_id': self.cognito_username, 'last':last}
        FunctionName = GLOBALS['LAMBDA']['get_project']

        for k in range(0, retry):
            try:
                response =  self.lambda_client.invoke(
                    FunctionName=FunctionName,
                    InvocationType= 'RequestResponse',
                    LogType='Tail',
                    # Payload=json.dumps(request),
                    Payload= bytes(json.dumps(request), encoding='utf8'),
                )
                # Get the serialized dataframe
                dict_str = response['Payload'].read().decode("UTF-8")
                if dict_str == 'null':
                    raise ValueError('Invalid lambda function response')
                data_dict = ast.literal_eval(dict_str)

                serialized_videos_data_dict = data_dict['body']['videos']
                data_df = pd.read_json(serialized_videos_data_dict)
                self.df = data_df

            except Exception as e:
                if 'expired' in str(e):
                    self.login() # try one more time
                else:
                    raise Exception('Lambda function error - {0}'.format(e))

        print("[pycollector.project]:  Returned %d videos" % len(self.df))        

        
    def __repr__(self):
        return str("<pycollector.project: program=%s, videos=%d>" % (self._programid,
                                                                     len(self)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, k):
        if isinstance(k, int):
            return Video(mp4url=self.df.iloc[k].raw_video_file_path, jsonurl=self.df.iloc[k].annotation_file_path)
        elif isinstance(k, slice):
            return [Video(mp4url=v, jsonurl=a) for (v,a) in zip(self.df.iloc[k].raw_video_file_path, self.df.iloc[k].annotation_file_path)]
        else:
            raise ValueError('Invalid index "%s"' % (str(k)))
            
    def videos(self):
        return sorted([v for v in self], key=lambda v: v.uploaded(), reverse=True)

    def last(self, n=1):
        assert len(self) >= n, "Invalid length (videos=%d < n=%d)" % (len(self), n)
        V = self.videos()[-n:]
        return V if n>1 else V[0]        
    
    def quicklookurls(self, outfile=None, display=True):
        """Generate a standalong HTML file containing the quicklook URLs for the current filtered project"""
        assert vipy.version.is_at_least("0.8.2")
        urls = [url for v in self for url in v.fetchjson().quicklookurls()]  # FIXME
        vipy.visualize.urls(urls, outfile=outfile, display=display)
        return urls


def search():
    return Project(since='2020-09-01')

def last(n=1):
    return Project(since='2020-09-01', last=n).last(n)    

            
