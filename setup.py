from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="Credit_Card_Default",
    version='0.0.1',
    author='SPS',
    author_email='s@gmail.com',
    install_requires=get_requirements(r'C:\Users\shprs\Downloads\Stats\CreditCard Default\requirements.txt'),
    packages=find_packages()

)