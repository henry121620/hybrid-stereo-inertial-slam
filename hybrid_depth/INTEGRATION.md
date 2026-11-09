# Integrating Hybrid-Depth Core Modules into Upstream Rover-SLAM

Because this lightweight companion repository omits upstream third-party libraries (`DBoW3`, `g2o`, `Sophus`), ONNX runtime weights, and DBoW vocabularies, follow these steps to assemble and build the full binary.

## 1. Clone Upstream Repositories

```bash
mkdir -p workspace/roverslam_ws workspace/igev_ws

# Rover-SLAM (tested base commit: 452a37a)
git clone https://github.com/zzzzxxxx111/Rover-SLAM.git workspace/roverslam_ws/Rover-SLAM

# IGEV++ (tested base commit: 099dec9)
git clone https://github.com/gangweiX/IGEV-plusplus.git workspace/igev_ws/IGEVpp
```

## 2. Overlay Our Core Files

Copy the modified front-end, depth loader, example entrypoints, and configs over `Rover-SLAM`:

```bash
cp hybrid_depth/include/RoverIgevDepth.h workspace/roverslam_ws/Rover-SLAM/include/
cp hybrid_depth/include/Frame.h          workspace/roverslam_ws/Rover-SLAM/include/
cp hybrid_depth/src/RoverIgevDepth.cc    workspace/roverslam_ws/Rover-SLAM/src/
cp hybrid_depth/src/Frame.cc             workspace/roverslam_ws/Rover-SLAM/src/

cp examples/stereo_inertial_euroc.cc     workspace/roverslam_ws/Rover-SLAM/Examples/Stereo-Inertial/
cp examples/stereo_inertial_zed_eth3d.cc workspace/roverslam_ws/Rover-SLAM/Examples/Stereo-Inertial/
cp configs/*.yaml                        workspace/roverslam_ws/Rover-SLAM/Examples/Stereo-Inertial/

cp depth_inference/euroc_igevpp_eth3d_depth.py workspace/igev_ws/IGEVpp/
```

## 3. Update `Rover-SLAM/CMakeLists.txt`

Add `src/RoverIgevDepth.cc` to the `Rover-SLAM` shared library sources and add the ZED2i executable target:

```cmake
# Under add_library(${PROJECT_NAME} SHARED ...)
src/Frame.cc
src/RoverIgevDepth.cc

# At the bottom of CMakeLists.txt
add_executable(stereo_inertial_zed_eth3d
        Examples/Stereo-Inertial/stereo_inertial_zed_eth3d.cc)
target_link_libraries(stereo_inertial_zed_eth3d ${PROJECT_NAME})
```

*(Note: inside Docker, ensure `ONNXRUNTIME_ROOT` points to `/opt/onnxruntime-gpu` and remove `-march=native` from `CMakeLists.txt` if targeting portable builds.)*

