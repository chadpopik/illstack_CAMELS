from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cpythontest.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)