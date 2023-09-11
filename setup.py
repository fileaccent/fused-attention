from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import os

workspace_dir = Path(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="fused_attn",
    ext_modules=[
        CUDAExtension(
            name="fused_attn",
            sources=[str(workspace_dir / "src" / "fused_attn_extention.cu")],
            include_dirs=[str(workspace_dir / "include")],
            extra_compile_args=[
                "-O3", 
                "-std=c++20", 
                "-I/opt/rocm/include",
                "-I/opt/rocm/hip/include",
                "--no-offload-arch=gfx1030",
                "--no-offload-arch=gfx900",
                "--no-offload-arch=gfx906",
                "--no-offload-arch=gfx908"
                ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)


# --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 --amdgpu-target=gfx90a --amdgpu-target=gfx1030