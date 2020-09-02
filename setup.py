import sys
from setuptools import setup, find_packages
from setuptools import setup as setup_alias

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
           'install_requires':["vipy","boto3","xmltodict","pandas","torch"],
           'keywords':['computer vision machine learning ML CV privacy video image'],
           'classifiers':["Programming Language :: Python :: 3",
                          "Operating System :: OS Independent",
                          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]}


setup(
    name='pycollector',
    author=d_setup['author'],
    author_email=d_setup['author_email'],
    version=d_setup['version'],
    namespace_packages=d_setup['namespace_packages'],
    packages=d_setup['packages'],
    description=d_setup['description'],
    long_description=d_setup['long_description'],
    long_description_content_type=d_setup['long_description_content_type'],
    url=d_setup['url'],
    download_url=d_setup['download_url'],
    install_requires=d_setup['install_requires'],
    keywords=d_setup['keywords'],
    classifiers=d_setup['classifiers']
)

setup_alias(
    name='visym-collector',
    author=d_setup['author'],
    author_email=d_setup['author_email'],
    version=d_setup['version'],
    namespace_packages=d_setup['namespace_packages'],
    packages=d_setup['packages'],
    description=d_setup['description'],
    long_description=d_setup['long_description'],
    long_description_content_type=d_setup['long_description_content_type'],
    url=d_setup['url'],
    download_url=d_setup['download_url'],
    install_requires=d_setup['install_requires'],
    keywords=d_setup['keywords'],
    classifiers=d_setup['classifiers']
)

