# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#FROM ubuntu:20.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Allow passing UID and GID as build arguments (defaults to 1000)
ARG UID=1000
ARG GID=1000

# setup environment
ENV TZ=US
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    ccache \
    clang-format \
    git \
    sudo \
    wget \
    tar \
    vim \
    libtiff-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install gcc 11
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update && apt-get install -y gcc-11 g++-11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Install cmake
RUN mkdir ~/temp
RUN cd ~/temp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz
RUN tar xvzf cmake-3.28.6-linux-x86_64.tar.gz --strip-components=1 -C /usr/local
RUN cd ..
RUN rm -rf temp

# Install Python apt dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python pip dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade \
    black \
    numpy \
    pytest \
    setuptools \
    twine \
    wheel

# Install C++ Dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libblosc-dev \
    libeigen3-dev \
    libc++-dev \
    libglu1-mesa-dev \
    python-dev \
    python-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install boost 1.80
RUN wget https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz
RUN tar -xf boost_1_80_0.tar.gz
RUN cd boost_1_80_0 && \
    ./bootstrap.sh && \
    ./b2 && \
    ./b2 install

# Install tbb 2022.0.0
RUN wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2022.0.0.tar.gz
RUN tar -xvzf v2022.0.0.tar.gz
RUN cd oneTBB-2022.0.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make -j4 install

RUN apt-get update && apt-get install -y libjemalloc-dev libgtk2.0-dev

# Install opencv
WORKDIR /
RUN apt-get update && apt-get install -y wget git ssh libfreetype6 libfreetype6-dev libharfbuzz-dev gdb
RUN git clone https://github.com/opencv/opencv_contrib.git --branch 4.6.0
RUN git clone https://github.com/opencv/opencv.git --branch 4.6.0
RUN cd /opencv
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_FREETYPE=OFF \
      -D WITH_GTK=ON \
      ../opencv && \
    make -j6 && \
    make install
    

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


# Install open3d 0.18.0
WORKDIR /
RUN apt-get update && apt-get install -y xorg-dev libxcb-shm0 libglu1-mesa-dev clang libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev libtbb-dev libosmesa6-dev libudev-dev autoconf libtool
RUN git clone -b v0.18.0 https://github.com/isl-org/Open3D.git
RUN cd Open3D && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j2 && \
    make install


# Create a group with the specified GID if it doesn't already exist
RUN groupadd -g $GID docker || true

# Create a user with the specified UID and assign to the group
RUN useradd -m -u $UID -g $GID -s /bin/bash docker && echo "docker:docker" | chpasswd && usermod -aG sudo docker


#RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Create a user with the specified UID and assign to the group
#RUN useradd -m -u $UID -g $GID -s /bin/bash docker_user  # User is named "docker_user"



ENV TERM xterm-256color
CMD ["bash", "-l"]

USER docker
RUN export PATH="/usr/local/cuda/bin:PATH"
CMD /bin/bash
