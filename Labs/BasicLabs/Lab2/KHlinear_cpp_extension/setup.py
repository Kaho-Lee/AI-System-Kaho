from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='KHlinear_cpp',
      ext_modules=[cpp_extension.CppExtension('KHlinear_cpp', ['KHlinear.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})