from setuptools import find_namespace_packages
from setuptools import find_packages, setup

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="demo_Xray",
    version="0.0.1",
    author="RS",
    author_email="ravibhattinkw@gmail.com",
    #install_requires=get_requirements("requirements_dev.txt"),
    package=find_packages()
)
