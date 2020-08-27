from setuptools import setup, find_packages

version = '0.0.1'

setup(
    name='pycollector',
    author='Visym Labs',
    author_email='info@visym.com',
    version=version,
    packages=find_packages(),
    description='Visym Collector',
    long_description="Visym Collector",
    long_description_content_type="text/markdown",
    url='https://github.com/visym/collector',
    download_url='https://github.com/visym/collector/archive/%s.tar.gz' % version,
    install_requires=[
        "vipy",
        "boto3",
        "xmltodict",
        "pandas"
    ],
    keywords=['vision', 'learning', 'ML', 'CV'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)
