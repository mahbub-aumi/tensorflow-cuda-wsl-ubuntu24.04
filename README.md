# tensorflow-cuda-wsl-ubuntu24.04
Install TensorFlow GPU on WSL2 Ubuntu 24.04 (Windows 11) | CUDA, cuDNN, TensorRT &amp; PyTorch

##################
Cuda and cuDNN:
https://www.tensorflow.org/install/source#gpu_support_2


Tensorrt Support or TensorFlow 2.16.1:
https://github.com/tensorflow/tensorflow/issues/61468
##################


##################
sudo apt update

sudo apt upgrade

sudo apt install build-essential

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh
##################


##################
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run

sudo sh cuda_12.1.1_530.30.02_linux.run


nano ~/.bashrc

export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc


sudo nano /etc/ld.so.conf

/usr/local/cuda-12.1/lib64

sudo ldconfig

echo $PATH

echo $LD_LIBRARY_PATH

sudo ldconfig -p | grep cuda

nvcc --version
##################

##################

https://developer.nvidia.com/rdp/cudnn-archive

##################


tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz

cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive


sudo cp include/cudnn*.h /usr/local/cuda-12.1/include

sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64

sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*

cd ..

ls -l /usr/local/cuda-12.1/lib64/libcudnn*


test cudnn:

nano test_cudnn.c

// test_cudnn.c
#include <cudnn.h>
#include <stdio.h>

int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN successfully initialized.\n");
    } else {
        printf("cuDNN initialization failed.\n");
    }
    cudnnDestroy(handle);
    return 0;
}


gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-12.1/include -L/usr/local/cuda-12.1/lib64 -lcudnn

./test_cudnn


##################

https://developer.nvidia.com/tensorrt/download

##################

tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz

sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1



nano ~/.bashrc

export PATH=/usr/local/cuda-12.1/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH

source ~/.bashrc



sudo rm /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn*.so.8

sudo ln -s /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.x.x /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8



conda create --name tf python=3.9

conda activate tf


python -m pip install tensorflow[and-cuda]

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


cd /usr/local/TensorRT-8.6.1/python


pip install tensorrt-8.6.1-cp39-none-linux_x86_64.whl

pip install tensorrt_dispatch-8.6.1-cp39-none-linux_x86_64.whl

pip install tensorrt_lean-8.6.1-cp39-none-linux_x86_64.whl


import tensorflow as tf

tf.config.list_physical_devices('GPU')


import tensorflow as tf
print(tf.__version__)



pip install jupyterlab

jupyter lab

