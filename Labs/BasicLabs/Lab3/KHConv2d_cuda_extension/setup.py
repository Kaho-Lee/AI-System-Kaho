from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KHConv2d_cuda',
    ext_modules=[
        CUDAExtension('KHConv2d_cuda', [
            'KHConv2d_cuda.cpp',
            'KHConv2d_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })