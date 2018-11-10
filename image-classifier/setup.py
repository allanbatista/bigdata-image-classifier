from setuptools import find_packages
from setuptools import setup

setup(
    name='bigdata-minos',
    description='this module training minos',
    version='0.0.1',
    install_requires=[
        'tensorflow == 1.10.0',
        'google-cloud-storage',
        'h5py',
        'scipy'
    ],
    packages=find_packages(),
    include_package_data=True
)