#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="mv_cuda",
    packages=['mv_cuda'],
    ext_modules=[
        CUDAExtension(
            name="mv_cuda._C",
            sources=[
            "mv_cuda.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc":  ["-Xptxas" ,"-O3","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm/")], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
