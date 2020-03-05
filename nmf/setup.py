from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "nmf",
    ext_modules = cythonize('nmf.pyx'), # accepts a glob pattern
    include_dirs=[numpy.get_include()]
)
