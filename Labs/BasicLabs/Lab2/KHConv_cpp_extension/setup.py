from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='KHConv_cpp',
      ext_modules=[cpp_extension.CppExtension('KHConv_cpp', ['KHConv.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})