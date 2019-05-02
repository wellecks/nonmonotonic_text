from setuptools import setup
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='tree_text_gen',
    version='0.0.1',
    description='tree text gen',
    packages=['tree_text_gen'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)
