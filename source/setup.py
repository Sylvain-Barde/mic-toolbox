#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:48:07 2018

@author: sb636
"""

import os
import sys
from setuptools import setup, Extension, find_packages
from distutils.errors import DistutilsModuleError

# Check for cython installation
try:
    from Cython.Distutils import build_ext as _build_ext
    HAVE_CYTHON = True
except ImportError:
    # As a fallback import the standard setuptools build_ext, and raise
    # error about Cython later
    from setuptools.command.build_ext import build_ext as _build_ext
    HAVE_CYTHON = False


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(extName, [extPath])


class build_ext(_build_ext):
    def initialize_options(self):
        if not HAVE_CYTHON:
            raise DistutilsModuleError(
                'Cython is required to compile the package.\n'
                'Cython can be obtained at www.cython.org or installed with '
                'conda or pip.')
        super(build_ext, self).initialize_options()

    def finalize_options(self):
        try:
            import numpy
        except ImportError:
            raise DistutilsModulesError('Building extension modules requires numpy')

        for ext in self.distribution.ext_modules:
            ext.include_dirs.extend([numpy.get_include(), '.'])
            ext.cython_directives = {
                "cdivision": True,
                "cdivision_warnings": False
            }

        super(build_ext, self).finalize_options()

setup(
  name="mic-toolbox",
  version="0.1.0a0",
  packages=find_packages(),
  ext_modules=[makeExtension(name) for name in scandir('mic')],
  cmdclass={'build_ext': build_ext},
  options = {'build_ext': {'inplace': True, 'force': True}}
)

