#!/usr/bin/env python3

import os
import setuptools
import subprocess
import sys
from glob import glob
from wheel.bdist_wheel import bdist_wheel

import distutils.sysconfig as sysconfig
import os
from distutils.sysconfig import get_python_inc
python_lib_location = os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY'))
python_include_dir = get_python_inc()

class platform_bdist_wheel(bdist_wheel):
    """Patched bdist_well to make sure wheels include platform tag."""
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False
"""
def _find_packages():
    packages = setuptools.find_packages()
    packages.append('mypythonpackage.doc')
    packages.append('matlabsources')
    print('packages found: {}'.format(packages))
    return packages
""" 

def configure_c_extension():
    """Configure cmake project to C extension."""
    print("Configuring for python {}.{}...".format(sys.version_info.major,
                                                   sys.version_info.minor))
    os.makedirs('cmake_build', exist_ok=True)
    cmake_command = [
        'cmake',
        '../',
        '-DPYTHON_EXECUTABLE=' + sys.executable,
	'-DPYTHON_LIBRARY=' + python_lib_location,
	'-DPYTHON_INCLUDE_DIR=' +  python_include_dir
    ]
    subprocess.check_call(cmake_command, cwd='cmake_build')


def build_c_extension():
    """Compile C extension."""
    print("Compiling extension...")
    subprocess.check_call(['make', '-j4'], cwd='cmake_build')


def create_package():
    subprocess.run(['mkdir', '-p', 'pytheia'], cwd='src')
    files = glob('cmake_build/lib/*.so')
    subprocess.run(['cp'] + files + ['src/pytheia'])
    subprocess.run(['touch', 'src/pytheia/__init__.py'])


create_package()
configure_c_extension()
build_c_extension()

setuptools.setup(
    name='pytheia',
    version='0.1.0',
    description='A Structure from Motion library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shengyu17/TheiaSfM/tree/feature/python_bindings',
    project_urls={
        "Documentation": "http://theia-sfm.org/",
    },
    author='Shengyu Yin',
    author_email = "shengyu952014@outlook.com",
    license='BSD',
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,

    package_dir={
        'pytheia': 'src/pytheia',
    },

    package_data={
        'pytheia': [
            'pytheia_sfm.*',
            'pytheia_image.*',
            'pytheia_io.*',
            'pytheia_solvers.*',
            'pytheia_matching.*',
            'libvlfeat.*',
            'libflann_cpp.*',

        ]
    },


    cmdclass={'bdist_wheel': platform_bdist_wheel},
)
