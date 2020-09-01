from setuptools import setup, find_packages, find_namespace_packages

d_version = {}
with open("./pycollector/version.py") as fp:
    exec(fp.read(), d_version)
version = d_version['VERSION']


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
# python3 -m pip install --upgrade setuptools wheel twine
# python3 setup.py sdist upload -r pypi
# ```


setup(
    name='pycollector',
    author='Visym Labs',
    author_email='info@visym.com',
    version=version,
    namespace_packages=['pycollector'],
    packages=find_packages(),
    description='Visym Collector',
    long_description="Visym Collector Python Tools for Live Visual Datasets",
    long_description_content_type="text/markdown",
    url='https://github.com/visym/collector',
    download_url='https://github.com/visym/collector/archive/%s.tar.gz' % version,
    install_requires=[
        "vipy",
        "boto3",
        "xmltodict",
        "pandas",
        "torch"
    ],
    keywords=['vision', 'learning', 'ML', 'CV'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
