# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:48:07 2018

@author: sb636
"""

import os
import sys
from distutils.core import setup
from distutils.extension import Extension
import numpy

# Check for cython installation
try:
    from Cython.Distutils import build_ext
except:
    print("Cython is required to compile the package.")
    print("Cython can be obtained at www.cython.org")
    sys.exit(1)
    


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
    return Extension(
        extName,
        [extPath],
        include_dirs = [numpy.get_include(),'.']
        )


extNames = scandir('mic')

extensions = [makeExtension(name) for name in extNames]

for ext in extensions:
    ext.cython_directives = {"cdivision": True,
                           "cdivision_warnings": False}

setup(
  name="mic toolbox",
  ext_modules=extensions,
  cmdclass = {'build_ext': build_ext},
  script_args = ['build_ext'],
  options = {'build_ext':{'inplace':True, 'force':True}}
)

