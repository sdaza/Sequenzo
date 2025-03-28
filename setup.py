"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : Sequenzo Package Setup Configuration

This file is maintained for backward compatibility and to handle C++ extension compilation.
Most configuration is now in pyproject.toml.
"""
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from Cython.Build import cythonize
import pybind11
import numpy
import os
import sys
from glob import glob


def get_extra_compile_args():
    if sys.platform == 'win32':
        return ['/std:c++14', '/EHsc', '/bigobj', '/O2', '/Gy']
    elif sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        return ['-std=c++11', '-Wall', '-Wextra']
    else:
        return ['-std=c++11', '-Wall', '-Wextra']


def get_cython_compile_args():
    extra_compile_args = []
    if sys.platform == "win32":
        extra_compile_args.append("/O2")
    # 不设置 macOS 或 Linux 的编译参数，让系统自动决定
    return extra_compile_args


def configure_cpp_extension():
    try:
        ext_module = Pybind11Extension(
            'sequenzo.dissimilarity_measures.c_code',
            sources=glob('sequenzo/dissimilarity_measures/src/*.cpp'),  # 👈 自动收集所有 .cpp 文件
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                'sequenzo/dissimilarity_measures/src/',
            ],
            extra_compile_args=get_extra_compile_args(),
            language='c++',
        )
        print("C++ extension configured successfully")
        return [ext_module]
    except Exception as e:
        print(f"Warning: Unable to configure C++ extension: {e}")
        print("The package will be installed with a Python fallback implementation.")
        return []


def configure_cython_extension():
    try:
        ext_module = Extension(
            name="sequenzo.clustering.utils.point_biserial",
            sources=["sequenzo/clustering/utils/point_biserial.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=get_cython_compile_args(),
        )
        print("Cython extension configured successfully")
        return cythonize(ext_module, compiler_directives={"language_level": "3"})
    except Exception as e:
        print(f"Warning: Unable to configure Cython extension: {e}")
        return []


# 自动创建目标目录，防止 copy .so 报错
os.makedirs("sequenzo/dissimilarity_measures/src", exist_ok=True)
os.makedirs("sequenzo/clustering/utils", exist_ok=True)


setup(
    ext_modules=configure_cpp_extension() + configure_cython_extension(),
    cmdclass={"build_ext": build_ext},
)