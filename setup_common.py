import sys
from setuptools import setup, find_packages

## Tag
#
# To create a tag in the repo
#
# ```bash
#     git tag X.Y.Z -m "pycollector-X.Y.Z"
#     git push --tags origin master
# ```
#
## PyPI distribution
#
# ```bash
# python3 setup.py sdist upload -r pypi
# ```

d_version = {}
with open("./pycollector/version.py") as fp:
    exec(fp.read(), d_version)
version = d_version['VERSION']

d_setup = {'author':'Visym Labs',
           'author_email':'info@visym.com',
           'version':version,
           'namespace_packages':['pycollector'],
           'packages':find_packages(),
           'description':'Visym Collector',
           'long_description':"Visym Collector Python Tools for Live Visual Datasets",
           'long_description_content_type':"text/markdown",
           'url':'https://github.com/visym/collector',
           'download_url':'https://github.com/visym/collector/archive/%s.tar.gz' % version,
           'install_requires':["vipy","dask","distributed","boto3","xmltodict","pandas","torch","torchvision","pytorch_lightning","ujson"],
           #'dependency_links':[]  # FIXME: there is an issue with windows installs of torch
           'keywords':['computer vision machine learning ML CV privacy video image'],
           'classifiers':["Programming Language :: Python :: 3",
                          "Operating System :: OS Independent",
                          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"],
           'include_package_data':True}
