from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mimical',

    version='0.2.7',

    description='Intensity modelling for multiply-imaged objects',

    long_description=long_description,

    author='Struan Stevenson',

    author_email='struan.stevenson@ed.ac.uk',

    packages= find_packages(),

    include_package_data=True,

    install_requires=["numpy", "astropy", "matplotlib", "nautilus-sampler", "petrofit", "tqdm", "pandas", "corner", "dynesty"],

    project_urls={
        "GitHub": "https://github.com/strusteve/mimical"
    }

)