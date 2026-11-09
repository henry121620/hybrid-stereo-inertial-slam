#ifndef ROVER_IGEV_DEPTH_H
#define ROVER_IGEV_DEPTH_H

#include <opencv2/core.hpp>

namespace ORB_SLAM3
{
namespace RoverIgevDepth
{
bool Enabled();
bool LoadDepthForTimestamp(const double timestampSec, cv::Mat& depth);
}
}

#endif
