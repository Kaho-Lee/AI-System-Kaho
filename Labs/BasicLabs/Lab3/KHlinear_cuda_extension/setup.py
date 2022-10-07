from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KHlinear_cuda',
    ext_modules=[
        CUDAExtension('KHlinear_cuda', [
            'KHlinear_cuda.cpp',
            'KHlinear_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })