FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=3.4.13
ARG PANGOLIN_VERSION=v0.6
ARG ONNXRUNTIME_VERSION=1.16.3

ENV TZ=Etc/UTC
ENV ROS_DISTRO=noetic
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime-gpu
ENV OpenCV_DIR=/usr/local/share/OpenCV
ENV LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg2 \
        libboost-all-dev \
        libcrypto++-dev \
        libeigen3-dev \
        libegl1-mesa-dev \
        libglew-dev \
        libgl1-mesa-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libgtk2.0-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libssl-dev \
        libtbb-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libv4l-dev \
        libwayland-dev \
        libxkbcommon-dev \
        libyaml-cpp-dev \
        nano \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        unzip \
        wget \
        xauth \
        zsh \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros1.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-rviz \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        numpy==1.23.5 \
        scipy \
        tqdm \
        matplotlib \
        opencv-python-headless \
        scikit-image \
        tensorboard \
        timm==0.5.4 \
        pyyaml \
        pandas

# IGEV++ original README uses torch 1.12.1 + cu113.  The CUDA 11.8 image can run
# cu113 wheels through NVIDIA driver compatibility and keeps this package close
# to the tested upstream IGEV++ dependency set.
RUN python3 -m pip install --no-cache-dir \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113

RUN wget -q "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip" -O /tmp/opencv.zip \
    && unzip -q /tmp/opencv.zip -d /tmp \
    && cmake -S "/tmp/opencv-${OPENCV_VERSION}" -B /tmp/opencv-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENGL=ON \
        -DWITH_QT=OFF \
    && cmake --build /tmp/opencv-build --parallel "$(nproc)" \
    && cmake --install /tmp/opencv-build \
    && ldconfig \
    && rm -rf /tmp/opencv.zip "/tmp/opencv-${OPENCV_VERSION}" /tmp/opencv-build

RUN wget -q "https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/${PANGOLIN_VERSION}.tar.gz" \
        -O /tmp/pangolin.tar.gz \
    && mkdir -p /tmp/pangolin-src \
    && tar -xzf /tmp/pangolin.tar.gz -C /tmp/pangolin-src --strip-components=1 \
    && cmake -S /tmp/pangolin-src -B /tmp/pangolin-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_PANGOLIN_PYTHON=OFF \
        -DBUILD_TESTS=OFF \
    && cmake --build /tmp/pangolin-build --parallel "$(nproc)" \
    && cmake --install /tmp/pangolin-build \
    && ldconfig \
    && rm -rf /tmp/pangolin.tar.gz /tmp/pangolin-src /tmp/pangolin-build

RUN wget -q \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
        -O /tmp/onnxruntime.tgz \
    && mkdir -p "${ONNXRUNTIME_ROOT}" \
    && tar -xzf /tmp/onnxruntime.tgz -C "${ONNXRUNTIME_ROOT}" --strip-components=1 \
    && test -f "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" \
    && test -f "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so" \
    && echo "${ONNXRUNTIME_ROOT}/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig \
    && rm -f /tmp/onnxruntime.tgz

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "export LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:/usr/local/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >> /root/.bashrc \
    && echo "source /opt/ros/noetic/setup.zsh" >> /root/.zshrc \
    && echo "export LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:/usr/local/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >> /root/.zshrc

WORKDIR /workspace

CMD ["/usr/bin/zsh"]
