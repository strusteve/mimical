import setuptools
from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mimical',

    version='0.1.5',

    description='Intensity modelling of multiply-imaged objects',

    long_description=long_description,

    author='Struan Stevenson',

    author_email='struan.stevenson@ed.ac.uk',

    packages= ["mimical", "mimical.fitting", "mimical.plotting", "mimical.utils",],

    include_package_data=True,

    install_requires=["numpy", "astropy", "matplotlib", "nautilus-sampler", "petrofit"],

)