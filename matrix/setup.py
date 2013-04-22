from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "nmf",
    ext_modules = cythonize('nmf.pyx'), # accepts a glob pattern
)
