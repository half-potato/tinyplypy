import sys
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# If pybind11 is not found, we raise an error
try:
    import pybind11
except ImportError:
    print("pybind11 not found. Please install it first: pip install pybind11")
    sys.exit(1)

# Optional: check for a specific version of pybind11
# from pkg_resources import parse_version
# if parse_version(pybind11.__version__) < parse_version('2.6.0'):
#     print("pybind11 >= 2.6.0 is required.")
#     sys.exit(1)

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()` 
    method can be invoked. 
    """
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        # The name of your resulting extension (.so on Linux, .pyd on Windows)
        "tinyplypy._tinyplypy_binding",
        sources=[
            "tinyplypy/bindings.cpp",  # your pybind11 bindings
            # If tinyply has its own .cpp, add it here:
            # "tinyplypy/tinyply.cpp"
        ],
        include_dirs=[
            "tinyplypy",                  # so #include "tinyply.h" works
            str(get_pybind_include())  # pybind11 headers
        ],
        # Set any extra compiler flags if you want. 
        # Example: c++11/14/17 standard, or warnings:
        extra_compile_args=["-std=c++11"],  
        language="c++"  # We're building a C++ extension
    ),
]

setup(
    name="tinyplypy",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A tinyply + pybind11 binding to read/write .ply files using NumPy arrays",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # If you want to package python modules (none here, just an example):
    # packages=["your_python_package"],
    # zip_safe=False,  # for pybind11 extensions
)

