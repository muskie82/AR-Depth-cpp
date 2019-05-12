# AR-Depth-cpp

### 1. Overview
<p align="center"> <img src="https://github.com/muskie82/AR-Depth-cpp/blob/master/gif/file.gif" width="500" height="320"> </p>

This is a C++ implementation of [Fast Depth Densification for Occlusion-aware Augmented Reality (SIGGRAPH ASIA2018)](https://homes.cs.washington.edu/~holynski/publications/occlusion/index.html) and [its python implementation](https://github.com/facebookresearch/AR-Depth).

The code generates dense depth map from sparse depth points of visual SLAM by using variational method.

Sample data is from [AR-Depth](https://github.com/facebookresearch/AR-Depth).

### 2. Installation
#### 2.1 Dependencies
1. OpenCV (>3.2)
2. Eigen 3.3.5
#### 2.3 Build

- Download the repository.

		git clone https://github.com/muskie82/AR-Depth-cpp.git

- Modify paths to include directories and libraries of TensorFlow in `CMakeLists.txt`.

- Build

		cd AR-Depth-cpp
		mkdir build
		cd build
		cmake ..
		make -j4
	

### 3 Usage
You can run sample program

		./ARDepth


### 4 Reference
* **Fast Depth Densification for Occlusion-aware Augmented Reality**, *Aleksander Holynski and Johannes Kopf*, IACM Transactions on Graphics (Proc. SIGGRAPH Asia). (https://homes.cs.washington.edu/~holynski/publications/occlusion/index.html)
* **AR-Depth**, (https://github.com/facebookresearch/AR-Depth)

### 5 License
GPLv3 license.
