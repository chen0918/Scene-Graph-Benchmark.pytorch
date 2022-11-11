## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python <= 3.8
- PyTorch >= 1.2 (Mine 1.4.0 (CUDA 10.1))
- torchvision >= 0.4 (Mine 0.5.0 (CUDA 10.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV

### **For[CUDA11.3, RTX 3090, 3080/ti, etc.]** Step-by-step installation
Reference: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/issues/148
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name scene_graph_benchmark
conda activate scene_graph_benchmark

conda install python=3.8
# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# Download and Install CUDA11.3
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

# install pytorch 1.10.0
conda install pytorch==1.10.0  torchvision==0.11.0 cudatoolkit==11.3.1 -c pytorch
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
# WARNING if you use older Versions of Pytorch (anything below 1.7), you will need a hard reset,
# as the newer version of apex does require newer pytorch versions. Ignore the hard reset otherwise.
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac

# WARNING set the environment variable to let the apex build with the compute compatibility of Ampere GPU (compute compatibility 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git

cd scene-graph-benchmark
# !!!!! Modify `maskrcnn_benchmark/utils/imports.py` line4: 'if torch._six.PY3' ->  'if torch._six.PY37'   

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
```

If error `ImportError: cannot import name 'container_abcs' from 'torch._six'`  happenes when runing the train progress, replace code ` from torch._six import container_abcs` to `import collections.abc as container_abcs`