from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
setup(name="inc",
  ext_modules=cythonize(
    [Extension("inc.algorithms.extract_paths_cython", ["inc/algorithms/extract_paths_cython.pyx"])],
    compiler_directives={'language_level' : "3"},
    annotate=False),
  include_dirs=[numpy.get_include()],
)