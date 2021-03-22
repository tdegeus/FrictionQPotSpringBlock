
from setuptools import setup, Extension

import re
import os
import pybind11
import pyxtensor
from os import environ
from setuptools_scm import get_version

version = get_version()

include_dirs = [
    os.path.abspath('include/'),
    pyxtensor.find_pyxtensor(),
    pyxtensor.find_pybind11(),
    pyxtensor.find_xtensor(),
    pyxtensor.find_xtl()]

build = pyxtensor.BuildExt

xsimd = pyxtensor.find_xsimd()

if xsimd:
    if len(xsimd) > 0:
        include_dirs += [xsimd]
        build.c_opts['unix'] += ['-march=native', '-DXTENSOR_USE_XSIMD']
        build.c_opts['msvc'] += ['/DXTENSOR_USE_XSIMD']

build.c_opts['unix'] += ['-DFRICTIONQPOTSPRINGBLOCK_VERSION="{0:s}"'.format(version)]
build.c_opts['msvc'] += ['/DFRICTIONQPOTSPRINGBLOCK_VERSION="{0:s}"'.format(version)]

ext_modules = [Extension(
    'FrictionQPotSpringBlock',
    ['python/main.cpp'],
    include_dirs = include_dirs,
    language = 'c++')]

setup(
    name = 'FrictionQPotSpringBlock',
    description = 'Spring-block friction model with local disordered potential energy landscape.',
    long_description = 'Spring-block friction model with local disordered potential energy landscape.',
    keywords = 'Friction',
    version = version,
    license = 'MIT',
    author = 'Tom de Geus',
    author_email = 'tom@geus.me',
    url = 'https://github.com/tdegeus/FrictionQPotSpringBlock',
    ext_modules = ext_modules,
    setup_requires = ['pybind11', 'pyxtensor'],
    cmdclass = {'build_ext': build},
    zip_safe = False)
