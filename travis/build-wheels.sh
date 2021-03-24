
#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
         auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}
#apt-get install
#curl https://raw.githubusercontent.com/dvershinin/apt-get-centos/master/apt-get.sh -o /usr/local/bin/apt-get
#chmod 0755 /usr/local/bin/apt-get

yum install -y wget
yum install -y gflags-devel
yum install -y atlas-devel
yum install -y eigen3-devel
yum install -y suitesparse-devel
#yum install -y libgoogle-glog-devel
cd /home
git clone https://github.com/google/glog.git
cd glog && mkdir build && cd build
cmake ../ && make && make install


#rocksdb dependencies
yum install -y centos-release-scl
yum install -y rh-python36
scl enable rh-python36 bash
yum install -y snappy snappy-devel
yum install -y zlib zlib-devel
yum install -y bzip2 bzip2-devel
yum install -y lz4-devel

cd /home
wget https://github.com/facebook/zstd/archive/v1.1.3.tar.gz
tar zxvf zstd-1.1.3.tar.gz
cd zstd-1.1.3
make && make install

#rocksdb
cd /home
git clone https://github.com/facebook/rocksdb.git
make -j4 shared_lib
make install

#rapidjson
cd /home
git clone https://github.com/Tencent/rapidjson.git
mv rapidjson/include/rapidjson /usr/include




#apt-get install cmake
# google-glog + gflags
#apt-get install libgflags-dev


#apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
#apt-get install libatlas-base-dev 




# Eigen3

#apt-get install libeigen3-dev 
# SuiteSparse and CXSparse (optional) 
#apt-get install libsuitesparse-dev



# openimageio
OIIO_VERSION="1.6.18"
yum install -y libpng-devel
#yum install -y libtiff-devel
git clone https://github.com/vadz/libtiff.git
cd libtiff
./configure
make && make install

git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
cd libjpeg-turbo/
cmake -G"Unix Makefiles" -DWITH_JPEG8=1 .
make && make install
yum install -y libjpeg-devel

cd /home
git clone https://github.com/AcademySoftwareFoundation/openexr

cd openexr
git checkout v2.5.5
mkdir build && cd build
cmake .. && make && make install

cd /home
git clone https://github.com/AcademySoftwareFoundation/Imath.git
cd Imath && mkdir build 
cd build && cmake .. && make && make install

#boost c++ library
cd /home
wget https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.gz
tar -xf boost_1_61_0.tar.gz
cd boost_1_61_0
./bootstrap.sh
./b2 install


#openimageio
cd /home
git clone https://github.com/ZJCRT/oiio && cd oiio && git checkout Release-$​​​​​​OIIO_VERSION​​​​​​​​ && mkdir -p release && cd release/ && cmake ../ -DSTOP_ON_WARNING=OFF -DUSE_OPENCV=0 -DOIIO_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release              -DUSE_PYTHON=OFF -DUSE_PYTHON3=OFF -DUSE_FFMPEG=OFF -DUSE_OPENGL=OFF              -DOIIO_BUILD_CPP11=ON -DCMAKE_CXX_FLAGS=-isystem\ /opt/libjpeg-turbo/include 

make -j​​​4 
make install && \
cd /home && \  
rm -fr oiio


# Build Ceres
CERES_VERSION="1.14.0"
cd /home
wget https://github.com/ceres-solver/ceres-solver/archive/$​​​​CERES_VERSION​​​​​​​​.zip &&  \
unzip $​​​​CERES_VERSION​​​​​​​​.zip && \
cd ceres-solver-$​​​​CERES_VERSION​​​​​​​​ && \
mkdir build && \
cd build && \
cmake  ../ \
-DCXX11=ON \
-DBUILD_DOCUMENTATION=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD-BENCHMARKS=OFF -DEIGENSPARSE=ON -DSUITESPARSE=OFF -DCMAKE_BUILD_TYPE=Release 
make -j​​​​4 && \
make install && \
cd /home && rm -rf $​​​​​​​​CERES_VERSION​​​​​​​​.zip && \
rm -rf ceres-solver-$​​​​​​​​CERES_VERSION​​​​​​​​



# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install pytheia --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" pytheia)
done
