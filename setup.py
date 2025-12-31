# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class CMakeBuild(_build_ext):
    """Custom build extension that uses CMake to build the extension."""

    def run(self):
        """Run CMake build process."""
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        """Build extension using CMake."""
        # Get absolute paths
        source_dir = Path(__file__).parent.absolute()
        build_temp = Path(self.build_temp).absolute()
        build_lib = Path(self.build_lib).absolute()

        # Create build directory
        build_temp.mkdir(parents=True, exist_ok=True)

        # Determine build type
        cfg = 'Debug' if self.debug else 'Release'

        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        # Build arguments
        build_args = ['--config', cfg]

        # Add parallel build flag
        if platform.system() != "Windows":
            build_args += ['--', f'-j{os.cpu_count() or 4}']
        else:
            build_args += ['--', f'/m:{os.cpu_count() or 4}']

        # Set CMAKE_BUILD_PARALLEL_LEVEL
        env = os.environ.copy()
        env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(os.cpu_count() or 4)

        # Run CMake configuration
        subprocess.check_call(
            ['cmake', str(source_dir)] + cmake_args,
            cwd=str(build_temp),
            env=env
        )

        # Run CMake build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=str(build_temp),
            env=env
        )


# The directory containing this file
HERE = Path(__file__).parent

# Read the version from pyproject.toml if needed
VERSION = "0.1.0"

# Define a dummy extension to trigger the build
ext_modules = [
    Extension(
        "_pypto_core",
        sources=[],  # CMake handles sources
    ),
]

setup(
    name="pypto",
    version=VERSION,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
)
