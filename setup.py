from distutils.core import setup
from io import open 
from setuptools import find_packages,setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyodds',
    version='1.0.0',
    description='An end-to-end anomaly detection system',
    author='Data Analytics at Texas A&M (DATA) Lab, Yuening Li',
    author_email='yuehningli@gmail.com',
    url='https://github.com/datamllab/PyODDS',
    install_requires=requirements,
    packages=find_packages(),
    python_requires='>=3.6',
)
