"""
CRAYNN - yet another NN toolkit.
"""

from setuptools import setup, find_packages
from codecs import open
import os.path as osp
import numpy as np

here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name = 'craynn',

  version='0.9.1',

  description="""Yet another neural network toolkit.""",

  long_description = long_description,

  url='https://github.com/maxim-borisyak/craynn',

  author='Maxim Borisyak and contributors.',
  author_email='mborisyak at hse dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at hse dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'Topic :: Scientific/Engineering :: Image Recognition',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
  ],

  keywords='neural networks',

  packages=find_packages(exclude=['contrib', 'examples', 'docs', 'tests']),

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'theano',
    'lasagne',
    'pydotplus',
  ],

  include_package_data=True,
  ext_modules = [],
  include_dirs = [np.get_include()]
)


