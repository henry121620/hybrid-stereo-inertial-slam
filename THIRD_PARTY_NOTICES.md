# Third-Party Notices and Attribution

This repository contains derivative components and interfaces built upon the following open-source projects. All original copyrights and licenses remain intact.

---

## 1. Rover-SLAM

* **Upstream Repository**: https://github.com/zzzzxxxx111/Rover-SLAM
* **Base Commit**: `452a37a`
* **License**: GNU General Public License v3.0 (GPL-3.0)
* **Modified/Extended Components**:
  * `hybrid_depth/include/Frame.h`, `hybrid_depth/src/Frame.cc`
  * `hybrid_depth/include/RoverIgevDepth.h`, `hybrid_depth/src/RoverIgevDepth.cc` (new)
  * `examples/stereo_inertial_euroc.cc`, `examples/stereo_inertial_zed_eth3d.cc`

## 2. ORB-SLAM3

* **Upstream Repository**: https://github.com/UZ-SLAMLab/ORB_SLAM3
* **License**: GNU General Public License v3.0 (GPL-3.0)
* **Copyright**: University of Zaragoza (Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M. M. Montiel, Juan D. Tardos, Raul Mur-Artal)
* **Note**: Rover-SLAM itself derives from ORB-SLAM3. Accordingly, all derivative C++ code in this repository is distributed under **GPL-3.0**.

## 3. IGEV++ (IGEV-plusplus)

* **Upstream Repository**: https://github.com/gangweiX/IGEV-plusplus
* **Base Commit**: `099dec9`
* **License**: MIT License
* **Copyright**: (c) 2024 Gangwei Xu
* **Components Referenced**: Core stereo depth estimation network (`IGEVStereo`) and pretrained `eth3d.pth` weights used via `depth_inference/euroc_igevpp_eth3d_depth.py`.

