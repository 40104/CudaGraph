from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "cpu_graph",                               
           sources=["cpu_graph.pyx", "graph.cpp"], 
           language="c++",
           extra_compile_args=['-fopenmp'],
           extra_link_args=['-fopenmp'],                      
      )))