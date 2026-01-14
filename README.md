# SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework

[![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat-square&logo=c%2B%2B&logoColor=white)](./src/slimvdb/slimvdb)
[![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](src/slimvdb/pybind)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/umfieldrobotics/slim-vdb/pulls)
[![Paper](https://img.shields.io/badge/paper-get-<COLOR>.svg?style=flat-square)](https://ieeexplore.ieee.org/document/11344775)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://lbesson.mit-license.org/)

![example](docs/overview.gif)

This is `SLIM-VDB`, a probabilistic semantic mapping framework built off of `VDBFusion` and `OpenVDB`. With this library, you can use either open-set or closed-set semantic labels with the same dataloaders. Semantic labels at each voxel are tracked probabilistically, mitigating "flickering" from uncertain network outputs. The highly efficient `OpenVDB` backend enables real-time, high-speed map integration and rendering.

## Installation

First clone the package to your machine. You will also need the OpenVDB library:
```
$ git clone -b slim-vdb git@github.com:umfieldrobotics/openvdb.git
$ git clone git@github.com:umfieldrobotics/slim-vdb.git
```

To use this package, you may build the docker environment by first [installing docker](https://docs.docker.com/engine/install/ubuntu/), and then running `docker build -t slimvdb_docker .` inside the `slim-vdb/docker/builder` directory. You will need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Then edit the docker `run.sh` file to mount the slim-vdb directory location as a volume (specifically, you need to edit the `-v $HOME:$HOME` and `-v /path/to/dataset:...` line to the location where you have cloned the SLIM-VDB/OpenVDB repos and the place you have downloaded your data. Start the docker by running `./run.sh`. If you have problems with opening display windows from within docker, try running `xhost +local:root` on your host machine.

### Build OpenVDB

Building OpenVDB for the first time might take a while, but you only will need to do this once!

1. `cd openvdb/build` (ensure you're on the `slim-vdb` branch)
2. `cmake -DOPENVDB_BUILD_PYTHON_MODULE=ON -DUSE_NUMPY=ON -DPYOPENVDB_INSTALL_DIRECTORY="/usr/local/lib/python3.9/dist-packages" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_ZLIB=OFF -DOPENVDB_BUILD_NANOVDB=ON -DNANOVDB_USE_CUDA=ON ..`
3. `make -j4 all install` (this might need to be run as `sudo`)

### Build SLIM-VDB

Many of SLIM-VDB's computational efficiencies come from compile-time code structure. In order to properly take advantage of this, there are compile-time parameters that must be set. 

1. `cd slim-vdb/build`
2. `cmake -DSLIMVDB_LANGUAGE=<OPEN or CLOSED> -DSLIMVDB_NCLASSES=<number of closed-set semantic classes or size of open-set latent space> ..` (for example, if you wanted to run closed-set mapping for scenenet you would set SLIMVDB_LANGUAGE to CLOSED and SLIMVDB_NCLASSES to 14. if you wanted to run open-set mapping you would set SLIMVDB_LANGUAGE to OPEN and SLIMVDB_NCLASSES to 512.)
3. `make install` (this might need to run as be `sudo`)
4. `cd ../examples/cpp/build`
5. `cmake ..`
6. `make all install`

After these steps are completed, there should be C++ executables under `slim-vdb/examples/cpp`. See the next section for an example on how to run them.

## Usage

Download a subset of the SceneNet dataset from [this link](https://drive.google.com/file/d/19joA_ZCmm_2_d86COT_rgPxqaX-5pbLf/view?usp=sharing), and place it in `slim-vdb/data/` directory.

The config files are found under `slim-vdb/examples/config`. You can change several parameters here, such as for the TSDF integration, mapping range, RGBD vs laser scan data input, realtime vs pre-computed segmentation, rendering image size, and the render color map.

### C++

To run the mapping executable, the parameter format is

```
./executable_name --config <path/to/config/file.yaml> --sequence <sequence number> <path/to/input/data/root> <path/to/output/directory>
```

As an example for SceneNet with the example data we provide:
```
./scenenet_pipeline --config config/scenenet.yaml --sequence 2 ../../data/ mesh_output
```

### Python

Instructions coming soon!

### Troubleshooting

Having any issues? Don't hesitate to open an issue and we can do our best to help you. 

## Realtime Segmentation

To enable realtime segmentation, set the flag in the config file (mentioned above). Additionally, you need to have compatible model weights downloaded and in the right location. Since the backbone of this code is in C++, the model weights need to be libtorch compatible. We provide an example set of weights from the [ESANet](https://github.com/TUI-NICR/ESANet) lightweight segmentation model in the data download link shared above. See the `RunInference` function inside `SceneNetOdometry.cpp` for more details. Note that the inference time of your model will greatly effect the overall runtime of SLIM-VDB.

## License

The [LICENSE-MIT](./LICENSE-MIT.txt) can be found at the root of this repository, which only applies to the code of `SLIM-VDB` but not to the adapted NanoVDB code (which uses [LICENSE-APACHE](./LICENSE-APACHE.txt)) or to [3rdparty dependencies](3rdparty/).

## Credits

I would like to thank the [VDBFusion](https://github.com/PRBonn/vdbfusion) and [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) authors for their well-written and documented code, which greatly aided in the development of `SLIM-VDB`.

## Citation

If you use this library, please cite the paper ([arxiv link](https://arxiv.org/pdf/2512.12945), [ieee link](https://ieeexplore.ieee.org/document/11344775)).

```bibtex
@article{sheppard2026slimvdb,
  author         = {Sheppard, Anja and Ewen, Parker and Wilson, Joey and Sethuraman, Advaith V. and Adewole, Benard and Li, Anran and Chen, Yuzhen and Vasudevan, Ram and Skinner, Katherine A.},
  title          = {SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework},
  journal        = {IEEE Robotics Automation and Letters},
  volume         = {TBD},
  year           = {2026},
  number         = {TBD},
  article-number = {TBD},
  url            = {https://ieeexplore.ieee.org/document/11344775},
  doi            = {10.1109/LRA.2026.3652875}
}
```
## Acknowledgment

This work was supported by the NSF under Grant DGE 224114 and AFOSR MURI under Grant FA9550-23-1-0400.

Additionally, we'd like to thank the authors of [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) and [VDBFusion](https://github.com/PRBonn/vdbfusion) for writing the well-documented code that SLIM-VDB was built on top of.
