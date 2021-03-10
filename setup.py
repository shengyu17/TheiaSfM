#!/usr/bin/env python3

import os
import setuptools
import subprocess
import sys
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

def _find_packages():
    packages = setuptools.find_packages()
    packages.append('mypythonpackage.doc')
    packages.append('matlabsources')
    print('packages found: {}'.format(packages))
    return packages
    

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
    author='Chris Sweeney',
    license='BSD',
    packages=['pytheia'],
    include_package_data=True,

    package_dir={
        'pytheia': 'cmake_build/lib',
    },

    #data_files=[('cmake_build', [ 'lib/a.txt', 'lib/pytheia_image.cpython-38-x86_64-linux-gnu.so','lib/pytheia_io.*','pytheia_matching.*', 'pytheia_sfm.*','pytheia_solvers.*','libflann_cpp*','libgtest*','libstatx*','libstlplus3*','libvisual_sfm.*','lib/libvlfeat.*']),    ],
                 
    

    # install_requires=[
    #     'cloudpickle>=0.4.0',
    #     'ExifRead>=2.1.2',
    #     'gpxpy>=1.1.2',
    #     'loky>=1.2.1',
    #     'networkx>=1.11',
    #     'numpy>=1.13',
    #     'pyproj>=1.9.5.1',
    #     'pytest>=3.0.7',
    #     'python-dateutil>=2.6.0',
    #     'PyYAML>=3.12',
    #     'scipy',
    #     'six',
    #     'xmltodict>=0.10.2',
    #     'Pillow>=6.0.0',
    # ],
    cmdclass={'bdist_wheel': platform_bdist_wheel},
)
