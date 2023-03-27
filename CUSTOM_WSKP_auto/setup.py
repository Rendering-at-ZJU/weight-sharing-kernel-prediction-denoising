from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CUSTOM_WSKP_auto',
    ext_modules=[
        CUDAExtension('CUSTOM_WSKP_auto', [
            'CUSTOM_WSKP_auto.cpp',
            'CUSTOM_WSKP_auto_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })