# Docker Files for DynoSAM

Base images are pulled from [docker-ros-ml-images](https://github.com/ika-rwth-aachen/docker-ros-ml-images)

- Dockerfile.amd64 is a linux/amd64 image tested on x86_64 desktop
- Dockerfile.l4t_jetpack6 is build from linux/arm64 tested on an NVIDIA ORIN NX with Jetpack 6

## Jetson Settings
Architecture | aarch64
Ubuntu | 22.04.5 LTS (Jammy Jellyfish)
Jetson Linux | 36.4.7
Python | 3.10.12
ROS | jazzy
CMake | 3.22.1
CUDA | 12.6.77-1
cuDNN | 9.3.0.75-1
TensorRT | 10.7.0.23-1+cuda12.6
PyTorch | 2.8.0
GPUs | (Orin (nvgpu))
OpenCV | 4.10.0

> NOTE: The CUDA/Pytorch/TensorRT versions settings come with the base dockerfile but in practice we have been using CUDA 12.9. 

## Other versioning
matplotlib=3.6.3
numpy=1.26.4

```
# Install GTSAM
RUN git clone https://github.com/borglab/gtsam.git
RUN cd gtsam && \
    git fetch && \
    git checkout tags/4.2.0 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_UNSTABLE=ON -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON -DGTSAM_TANGENT_PREINTEGRATION=OFF \
    -DGTSAM_BUILD_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -DCMAKE_BUILD_RPATH=/usr/lib/x86_64-linux-gnu \
    -DCMAKE_INSTALL_RPATH=/usr/lib/x86_64-linux-gnu \
    .. && \
    sed -i '176d' /root/gtsam/build/python/navigation.cpp && \
    make -j$(nproc) install && \
    python3 -c "import sysconfig; import shutil; import os; site_packages = sysconfig.get_path('purelib'); gtsam_python = '/root/gtsam/build/python'; if os.path.exists(gtsam_python + '/gtsam'): shutil.copytree(gtsam_python + '/gtsam', site_packages + '/gtsam', dirs_exist_ok=True); shutil.copytree(gtsam_python + '/gtsam_unstable', site_packages + '/gtsam_unstable', dirs_exist_ok=True); shutil.copytree('/root/gtsam/python/gtsam', site_packages + '/gtsam', dirs_exist_ok=True)"



```
```
sudo sed -i 's/attributes\.memoryType/attributes.type/g' /usr/local/include/pangolin/image/memcpy.h
```
