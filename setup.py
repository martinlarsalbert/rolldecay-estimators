#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('rolldecayestimators', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'rolldecay-estimators'
DESCRIPTION = 'A template for scikit-learn compatible packages.'
with open("README.md", "r") as fh:
    long_description = fh.read()

MAINTAINER = 'Martin Alexandersson'
MAINTAINER_EMAIL = 'maralex@chalmers.se'
URL = 'https://github.com/scikit-learn-contrib/project-template'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/martinlarsalbert/rolldecay-estimators'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn','pandas','sympy','matplotlib','dill']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

package_data= {
          "rolldecayestimators": ["*.csv"],
      }

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=long_description,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      package_data=package_data,
      author="Martin Alexandersson",
      author_email='maralex@chalmers.se',
      python_requires='>=3.5',
      keywords='rolldecayestimators',
      )
