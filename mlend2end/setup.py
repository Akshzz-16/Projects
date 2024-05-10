# this will help us to create this ml application as package as use anywhere after downloading. 

from setuptools import find_packages, setup
from typing import List


def get_requirements(filePath)->List[str]:
    '''
    this func will return list of requirements
    '''

    requirements = []
    hypen_e = '-e .'
    
    with open(filePath) as fileObj:
        requirements = fileObj.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]
        
        if hypen_e in requirements:
            requirements.remove(hypen_e)    

    return requirements


setup(
name='MLPRO',
version='0.0.1',
author='akash',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)