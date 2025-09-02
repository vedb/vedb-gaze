from setuptools import setup, find_packages

requirements = []

setup(
    name="vedb_gaze",
    version="0.0.0",
    packages=find_packages(),
    long_description=open("README.md").read(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'vedb_gaze':[
            'defaults.cfg',
            'externals/',
            'config/*yaml'
        ]
    }
)
