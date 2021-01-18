import sys
from setuptools import setup, find_packages
from setup_common import d_setup

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
    classifiers=d_setup['classifiers'],
    include_package_data=True
)


