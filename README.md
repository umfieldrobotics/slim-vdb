# SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework

[![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat-square&logo=c%2B%2B&logoColor=white)](./src/slimvdb/slimvdb)
[![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](src/slimvdb/pybind)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/umfieldrobotics/slim-vdb/pulls)
[![Paper](https://img.shields.io/badge/paper-get-<COLOR>.svg?style=flat-square)](TBD)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://lbesson.mit-license.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?style=flat-square)](https://colab.research.google.com/github/umfieldrobotics/slim-vdb/blob/main/examples/notebooks/kitti_odometry.ipynb)

![example](docs/overview.gif)

This is `SLIM-VDB`, a probabilistic semantic mapping framework built off of `VDBFusion` and `OpenVDB`. With this library, you can use either open-set or closed-set semantic labels with the same dataloaders. Semantic labels at each voxel are tracked probabilistically, mitigating "flickering" from uncertain network outputs. The highly efficient `OpenVDB` backend enables real-time, high-speed map integration and rendering.

## Installation

Coming soon...

## Usage

Coming soon...

## LICENSE

The [LICENSE-MIT](./LICENSE-MIT.txt) can be found at the root of this repository, which only applies to the code of `SLIM-VDB` but not to the adapted NanoVDB code (which uses [LICENSE-APACHE](./LICENSE-APACHE.txt)) or to [3rdparty dependencies](3rdparty/).

## Credits

I would like to thank the [VDBFusion](https://github.com/PRBonn/vdbfusion) and [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) authors for their well-written and documented code, which greatly aided in the development of `SLIM-VDB`.

## Citation

If you use this library, please cite the paper (link TBA).

```bibtex
@article{sheppard2025slimvdb,
  author         = {Sheppard, Anja and Ewen, Parker and Wilson, Joey and Sethuraman, Advaith V. and Adewole, Benard and Li, Anran and Chen, Yuzhen and Vasudevan, Ram and Skinner, Katherine A.},
  title          = {SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework},
  journal        = {Robotics Automation and Letters},
  volume         = {TBD},
  year           = {2025},
  number         = {TBD},
  article-number = {TBD},
  url            = {TBD},
  doi            = {TBD}
}
```
