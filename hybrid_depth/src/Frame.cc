/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"
#include "Extractors/SPextractor.h"
#include "Matchers/SPmatcher.h"
#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include "RoverIgevDepth.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <limits>

namespace ORB_SLAM3
{

namespace
{
bool RoverZedFrameDebugEnabled()
{
    const char* env = std::getenv("ROVER_ZED_DEBUG_FRAME");
    if (!env) return false;
    const std::string v(env);
    return !(v == "0" || v == "false" || v == "False" ||
             v == "FALSE" || v == "off" || v == "OFF");
}

int CountValidStereoDepth(const std::vector<float>& depths)
{
    int count = 0;
    for (const float z : depths)
    {
        if (std::isfinite(z) && z > 0.0f)
            count++;
    }
    return count;
}
}

// 下一个生成的帧的ID,这里是初始化类的静态成员变量
long unsigned int Frame::nNextId=0;

// 是否要进行初始化操作的标志
// 这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpSPvocabulary(frame.mpSPvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false),imgLeft(frame.imgLeft)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}

// 立体匹配模式下的双目
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, SPextractor* extractorLeft, SPextractor* extractorRight, SPVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpSPvocabulary(voc),mpExtractorLeft(extractorLeft),mpExtractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false), imgLeft(imLeft), imgRight(imRight)
{
    // Frame ID
    // Step 1 帧的ID 自增
    mnId=nNextId++;

    // Scale Level Info
    // Step 2 计算图像金字塔的参数 
	// 获取图像金字塔的层数
    mnScaleLevels = mpExtractorLeft->GetLevels();
    // // 获得层与层之间的缩放比
    mfScaleFactor = mpExtractorLeft->GetScaleFactor();
    // // 计算上面缩放比的对数
    mfLogScaleFactor = log(mfScaleFactor);
    // // 获取每层图像的缩放因子
    mvScaleFactors = mpExtractorLeft->GetScaleFactors();
    // // 同样获取每层图像缩放因子的倒数
    mvInvScaleFactors = mpExtractorLeft->GetInverseScaleFactors();
    // // 高斯模糊的时候，使用的方差
    mvLevelSigma2 = mpExtractorLeft->GetScaleSigmaSquares();
    // for(auto& i: mvLevelSigma2){
    //     cout<<i<<" ";
    // }
    // cout<<endl;
    // // 获取sigma^2的倒数
    mvInvLevelSigma2 = mpExtractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    //cout<<imLeft.cols<<endl;
    // Step 3 对左目右目图像提取ORB特征点, 第一个参数0-左图， 1-右图。为加速计算，同时开了两个线程计算
    thread threadLeft(&Frame::ExtractKeyPoints,this,0,imLeft,0,0);
    // 对右目图像提取orb特征
    thread threadRight(&Frame::ExtractKeyPoints,this,1,imRight,0,0);
    // 等待两张图像特征点提取过程完成
    threadLeft.join();
    threadRight.join();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // mvKeys中保存的是左图像中的特征点，这里是获取左侧图像中特征点的个数
    N = mvKeys.size();

    if (RoverZedFrameDebugEnabled())
    {
        std::cout << "[ZED FRAME DEBUG] ctor=pinhole_stereo_after_extract"
                  << " ts=" << std::fixed << std::setprecision(9) << mTimeStamp
                  << " N=" << N
                  << " left_keys=" << mvKeys.size()
                  << " right_keys=" << mvKeysRight.size()
                  << " left_desc_rows=" << mDescriptors.rows
                  << " right_desc_rows=" << mDescriptorsRight.rows
                  << std::endl;
    }

    // 如果左图像中没有成功提取到特征点那么就返回，也意味这这一帧的图像无法使用
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif

    // Step 5 计算双目间特征点的匹配，只有匹配成功的特征点会计算其深度,深度存放在 mvDepth 
	// mvuRight中存储的应该是左图像中的点所匹配的在右图像中的点的横坐标（纵坐标相同）
    ComputeStereoMatches();
    ApplyIgevDepthIfAvailable();
    if (RoverZedFrameDebugEnabled())
    {
        std::cout << "[ZED FRAME DEBUG] ctor=pinhole_stereo"
                  << " ts=" << std::fixed << std::setprecision(9) << mTimeStamp
                  << " N=" << N
                  << " left_keys=" << mvKeys.size()
                  << " right_keys=" << mvKeysRight.size()
                  << " stereo_valid=" << CountValidStereoDepth(mvDepth)
                  << std::endl;
    }
    //std::cout<<"s4.6"<<endl;
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imLeft);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

    // 双目相机基线长度
    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // Step 6 将特征点分配到图像网格中
    // 上个版本这句话放在了new 锁那个上面，放在目前这个位置更合理，因为要把一些当前模式不用的参数赋值，函数里面要用
    AssignFeaturesToGrid();
}

// RGBD
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    // Step 1 帧的ID 自增
    mnId=nNextId++;

    // Scale Level Info
    // Step 2 计算图像金字塔的参数 
	// 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    // 获得层与层之间的缩放比
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // 计算上面缩放比的对数
    mfLogScaleFactor = log(mfScaleFactor);
    // 获取每层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // 同样获取每层图像缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // 高斯模糊的时候，使用的方差
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // 获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // 获取特征点的个数
    N = mvKeys.size();

    // 如果这一帧没有能够提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
    UndistortKeyPoints();

    // Step 5 获取图像的深度，并且根据这个深度推算其右图中匹配的特征点的视差
    ComputeStereoFromRGBD(imDepth);

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    // 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

    // 计算假想的基线长度 baseline= mbf/fx
    // 后面要对从RGBD相机输入的特征点,结合相机基线长度,焦距,以及点的深度等信息来计算其在假想的"右侧图像"上的匹配点
    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // 将特征点分配到图像网格中
    AssignFeaturesToGrid();
}

// 单目模式
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, SPextractor* extractor,SPVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpSPvocabulary(voc),mpExtractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    // Step 1 帧的ID 自增
    imgLeft = imGray.clone();//后面考虑删除！
    cv::Mat img = imGray.clone();
    cv::Size frameSize = imGray.size();
    mnId=nNextId++;

    // Step 2 计算图像金字塔的参数 
    // Scale Level Info
	//获取图像金字塔的层数
    // mnScaleLevels = mpORBextractorLeft->GetLevels();
	// //获取每层的缩放因子
    // mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	// //计算每层缩放因子的自然对数
    // mfLogScaleFactor = log(mfScaleFactor);
	// //获取各层图像的缩放因子
    // mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	// //获取各层图像的缩放因子的倒数
    // mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	// //获取sigma^2
    // mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	// //获取sigma^2的倒数
    // mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    mnScaleLevels = mpExtractorLeft->GetLevels();
    mfScaleFactor = mpExtractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpExtractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpExtractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpExtractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpExtractorLeft->GetInverseScaleSigmaSquares();
    
    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

    // Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图
    //ExtractORB(0,imGray,0,1000);
    ExtractKeyPoints(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // 提取特征点的个数
    N = mvKeys.size();
    //cout<<"N: "<<N<<endl;
    // 如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正 
    UndistortKeyPoints();

    // Set no stereo information
    // 由于单目相机无法直接获得立体信息，所以这里要给右图像对应点和深度赋值-1表示没有相关信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    // 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

    //计算 basline
    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // 将特征点分配到图像网格中 
    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}

/** 
 * @brief 特征分网格
 */
void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    // Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    // 开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }

    // Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];
        // 存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                // 如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

/** 
 * @brief 赋值新的偏置
 * @param flag 左右标志位
 * @param im 图片
 * @param x0 界限
 * @param x1 界限
 */
void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    // 判断是左图还是右图
    if(flag==0)
        // 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中 
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        // 右图的话就需要使用右图指定的特征点提取器，并将提取结果保存到对应的变量中 
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

void Frame::ExtractKeyPoints(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0){
        monoLeft = (*mpExtractorLeft)(im,mvKeys,mDescriptors);

        //画出特征点图
        // for (const cv::KeyPoint& point : mvKeys) {
        //     cv::circle(im,point.pt,2,cv::Scalar(0, 0, 255), 0.5);
        // }
        // cv::imshow("Feature Points", im);
        // cv::waitKey(0);
    }
    else
        monoRight = (*mpExtractorRight)(im,mvKeysRight,mDescriptorsRight);
}

bool Frame::isSet() const {
    return mbIsSet;
}

/** 
 * @brief 赋值位姿
 * @param Tcw 位姿
 */
void Frame::SetPose(const Sophus::SE3<float> &Tcw)
{
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

/** 
 * @brief 赋值新的偏置
 * @param b 偏置
 */
void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

/** 
 * @brief 赋值新的速度
 * @param Vwb 速度
 */
void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

/** 
 * @brief 获取速度
 */
Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

/** 
 * @brief 赋值位姿与速度
 */
void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

/** 
 * @brief 通过mTcw更新其他关于位姿的变量
 */
void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

/** 
 * @brief 获得imu的平移
 */
Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

/** 
 * @brief 获得imu的旋转
 */
Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

/** 
 * @brief 获得imu的位姿
 */
Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

/** 
 * @brief 获得左右目的相对位姿
 */
Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

/** 
 * @brief 获得左右目的相对位姿
 */
Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

/** 
 * @brief 获得左右目的相对旋转
 */
Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

/** 
 * @brief 获得左右目的相对平移
 */
Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}

/**
 * @brief 判断路标点是否在视野中
 * 步骤
 * Step 1 获得这个地图点的世界坐标
 * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
 * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
 * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
 * Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于设定阈值，返回false
 * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
 * Step 7 记录计算得到的一些参数
 * @param[in] pMP                       当前地图点
 * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
 * @return true                         地图点合格，且在视野内
 * @return false                        地图点不合格，抛弃
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    // 单目，立体匹配双目，rgbd
    if(Nleft == -1)
    {
        // cout << "\na";
		// mbTrackInView是决定一个地图点是否进行重投影的标志
    	// 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        // Step 1 获得这个地图点的世界坐标
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        // Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，直接返回false
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        // Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
        // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        // Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
     	// 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
    	//  mOw：当前相机光心在世界坐标系下坐标
        const Eigen::Vector3f PO = P - mOw;
        // 取模就得到了距离
        const float dist = PO.norm();

        // 如果不在允许的尺度变化范围内，认为重投影不可靠
        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        // Step 5 关卡四：
        // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值,
        // 若小于cos(viewingCosLimit), 即夹角大于viewingCosLimit弧度则返回
        Eigen::Vector3f Pn = pMP->GetNormal();

        // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
        const float viewCos = PO.dot(Pn)/dist;

        // 如果大于给定的阈值 cos(60°)=0.5，认为这个点方向太偏了，重投影不可靠，返回false
        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        // Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Step 7 记录计算得到的一些参数
        // Data used by the tracking
    	// 通过置位标记 MapPoint::mbTrackInView 来表示这个地图点要被投影 
        pMP->mbTrackInView = true;
        // 该地图点投影在当前图像（一般是左图）的像素横坐标
        pMP->mTrackProjX = uv(0);
        // bf/z其实是视差，相减得到右图（如有）中对应点的横坐标
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        // 该地图点投影在当前图像（一般是左图）的像素纵坐标
        pMP->mTrackProjY = uv(1);
        // 根据地图点到光心距离，预测的该地图点的尺度层级
        pMP->mnTrackScaleLevel= nPredictedLevel;
        // 保存当前视角和法线夹角的余弦值
        pMP->mTrackViewCos = viewCos;

        // 执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
        return true;
    }
    // 左右目时分别验证
    else
    {
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

/** 
 * @brief 暂没用到
 */
bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

/** 
 * @brief 暂没用到
 */
Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

/** 
 * @brief 给定一个坐标，返回区域内所有特征点
 * @param x 给定点的x
 * @param y 给定点的y
 * @param r 搜索半径
 * @param minLevel 金字塔下边界
 * @param maxLevel 金字塔上边界
 * @param bRight 是否是右相机
 */
vector<size_t> Frame::GetFeaturesInArea(
    const float &x, const float  &y, const float  &r,
    const int minLevel,const int maxLevel, const bool bRight) const
{
    // 存储搜索结果的vector
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    // Step 1 计算半径为r圆左右上下边界所在的网格列和行的id
    // 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
    // (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
    // mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
	// (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
	// 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
    // 保证nMinCellX 结果大于等于0
    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    // 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    // 计算圆所在的右边界网格列索引
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    // 如果计算出的圆右边界所在的网格不合法，说明该特征点不好，直接返回空vector
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    // 后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id
    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    // 检查需要搜索的图像金字塔层数范围是否符合要求
    //? 疑似bug。(minLevel>0) 后面条件 (maxLevel>=0)肯定成立
    //? 改为 const bool bCheckLevels = (minLevel>=0) || (maxLevel>=0);
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            // 如果这个网格中没有特征点，那么跳过这个网格继续下一个
            if(vCell.empty())
                continue;

            // 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // 根据索引先读取这个特征点 
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    // cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
					// 保证特征点是在金字塔层级minLevel和maxLevel之间，不是的话跳过
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 如果x方向和y方向的距离都在指定的半径之内，存储其index为候选特征点
                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/**
 * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
 * 
 * @param[in] kp                    给定的特征点
 * @param[in & out] posX            特征点所在网格坐标的横坐标
 * @param[in & out] posY            特征点所在网格坐标的纵坐标
 * @return true                     如果找到特征点所在的网格坐标，返回true
 * @return false                    没找到返回false
 */
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // 计算特征点x,y坐标落在哪个网格内，网格坐标为posX，posY
    // mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
    // mfGridElementHeightInv=(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 因为特征点进行了去畸变，而且前面计算是round取整，所以有可能得到的点落在图像网格坐标外面
    // 如果网格坐标posX，posY超出了[0,FRAME_GRID_COLS] 和[0,FRAME_GRID_ROWS]，表示该特征点没有对应网格坐标，返回false
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    // 计算成功返回true
    return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 * 
 */
void Frame::ComputeBoW()
{
    // 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
        // 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        mpORBvocabulary->transform(vCurrentDesc,    //当前的描述子vector
                                   mBowVec,         //输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                                   mFeatVec,        //输出，记录node id及其对应的图像 feature对应的索引
                                   4);              //4表示从叶节点向前数的层数
    }
}
void Frame::binarize_descriptors(){
    mDescriptors_bin = cv::Mat(mDescriptors.size(), CV_8UC1);
    for(size_t i=0 ; i<mDescriptors.rows; ++i){
        cv::Mat tmp;
        cv::threshold(mDescriptors.row(i), tmp, 0, 1, cv::THRESH_BINARY);
        for(size_t j=0; j<mDescriptors.cols; j++){
            mDescriptors_bin.at<uchar>(i,j) = tmp.at<float>(j);
        }
    }
}
void Frame::ComputeBoW3()
{

    binarize_descriptors();
    if(mBow3Vec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors_bin);
        mpSPvocabulary->transform(vCurrentDesc,mBow3Vec,mFeat3Vec,0);  // 5 is better
    }

}

/**
 * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
 * 
 */
void Frame::UndistortKeyPoints()
{
    // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
	// 变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    // N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N,2,CV_32F);

    // 遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<N; i++)
    {
        // 然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    // cv::undistortPoints最后一个矩阵为空矩阵时，得到的点为归一化坐标点
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    // 调整回只有一个通道，回归我们正常的处理方式
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    // Step 3 存储校正后的特征点
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        // 根据索引获取这个特征点
		// 注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
        // 读取校正后的坐标并覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

/**
 * @brief 计算去畸变图像的边界
 * 
 * @param[in] imLeft            需要计算边界的图像
 */
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        // 校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));  // 左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));  // 右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));  // 左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));  // 左下和右下纵坐标最小的
    }
    else
    {
        // 如果畸变参数为0，就直接获得图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/*
 * 双目匹配函数
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n ‘
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n 
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一个单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就越准确
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    /*两帧图像稀疏立体匹配（即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
     * 输入：两帧立体矫正后的图像img_left 和 img_right 对应的orb特征点集
     * 过程：
          1. 行特征点统计. 统计img_right每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
          2. 粗匹配. 根据步骤1的结果，对img_left第i行的orb特征点pi，在img_right的第i行上的orb特征点集中搜索相似orb特征点, 得到qi
          3. 精确匹配. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
          4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取float精度的真实视差
          5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点。
          6. 删除离缺点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
     * 输出：稀疏特征点视差图/深度图（亚像素精度）mvDepth 匹配结果 mvuRight
     */

    // 为匹配结果预先分配内存，数据类型为float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    // orb特征相似度阈值  -> mean ～= (max  + min) / 2
    const float thOrbDist = (SPmatcher::TH_HIGH+SPmatcher::TH_LOW)/2;
    
    // 金字塔顶层（0层）图像高 nRows
    const int nRows = imgLeft.rows;
    //const int nRows = mpExtractorLeft->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    // 二维vector存储每一行的orb特征点的列坐标的索引，为什么是vector，因为每一行的特征点有可能不一样，例如
    // vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
    // vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    // 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();

    // Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for(int iR=0; iR<Nr; iR++)
    {
        // 获取特征点ir的y坐标，即行号
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
        // 2 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);
         
        // 将特征点ir保证在可能的行号中
        for(int yi=minr;yi<=maxr;yi++)
        {
            
            vRowIndices[yi].push_back(iR);
            
        }
            
    }
    
    // Step 2 -> 3. 粗匹配 + 精匹配
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ
    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;
    
    // For each left keypoint search a match in the right image
    // 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        // 计算理论上的最佳搜索范围
        const float minU = uL-maxD;
        const float maxU = uL-minD;

        // 最大搜索范围小于0，说明无匹配点
        if(maxU<0)
            continue;

        // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        float bestDist = SPmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;

            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if(uR>=minU && uR<=maxU)
            {
                // 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const float dist = SPmatcher::DescriptorDistance_sp(dL,dR);

                // 统计最小相似度及其对应的列坐标(x)
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }
        //cout<<"s5.30"<<endl;
        // Subpixel match by correlation
        // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        // Step 3. 精确匹配. 
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            // 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            // 滑动窗口搜索, 类似模版卷积或滤波
            // w表示sad相似度的窗口半径
            const int w = 5;
            // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
            //cout<<"kpL.octave: "<<kpL.octave<<"scaledvL-w"<<scaledvL-w<<"scaleduL-w: "<<scaleduL-w<<endl;
            //cv::Mat IL = mpExtractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            cv::Mat IL;
            try{
                IL = imgLeft.rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            }catch(const cv::Exception& e){
                cout<<"kpL.octave: "<<kpL.octave<<"scaledvL-w"<<scaledvL-w<<"scaleduL-w: "<<scaleduL-w<<endl;
            }
            
            
            // 初始化最佳相似度
            float bestDist = INT_MAX;
			// 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;
			// 滑动窗口的滑动范围为（-L, L）
            const int L = 5;
			// 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
            // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0-L-w;
            const float endu = scaleduR0+L+w+1;
            // 判断搜索是否越界
            //if(iniu<0 || endu >= mpExtractorRight->mvImagePyramid[kpL.octave].cols)
            if(iniu<0 || endu >= imgRight.cols)
            {
                //cout<<"s1.1"<<endl;
                continue;
            }
                
            // 在搜索范围内从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++)
            {   //cout<<scaleduR0<<endl;
                //cout<<scaledvL-w<<" "<<scaledvL+w+1<<" "<<scaleduR0+incR-w<<" "<<scaleduR0+incR+w+1<<endl;
                // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR;
                try
                {
                    IR = imgRight.rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                }
                catch(const cv::Exception& e)
                {
                   cout<<"kpL.octave: "<<kpL.octave<<"scaledvL-w"<<scaledvL-w<<"scaleduL-w: "<<scaleduL-w<<endl;
                }
                
                
                
                // sad 计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                // 统计最小sad和偏移量
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                // L+incR 为refine后的匹配点列坐标(x)
                vDists[L+incR] = dist;
            }

            // 搜索窗口越界判断ß 
            if(bestincR==-L || bestincR==L)
                continue;

            // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              . 
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                // 如果存在负视差，则约束为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 根据视差值计算深度信息
                // 保存最相似点的列坐标(x)信息
                // 保存归一化sad最小相似度
                // Step 5. 最优视差值/深度选择.
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }
    //cout<<"s5.3"<<endl;
    // Step 6. 删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            // 误匹配点置为-1，和初始化时保持一直，作为error code
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// 计算RGBD图像的立体深度信息
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    /** 主要步骤如下:.对于彩色图像中的每一个特征点:<ul>  */
    // mvDepth直接由depth图像读取`
	// 这里是初始化这两个存储“右图”匹配特征点横坐标和存储特征点深度值的vector
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        /** <li> 从<b>未矫正的特征点</b>提供的坐标来读取深度图像拿到这个点的深度数据 </li> */
		// 获取校正前和校正后的特征点
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        // 获取其横纵坐标，注意 NOTICE 是校正前的特征点的
        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        // 从深度图像中获取这个特征点对应的深度点
        // NOTE 从这里看对深度图像进行去畸变处理是没有必要的,我们依旧可以直接通过未矫正的特征点的坐标来直接拿到深度数据
        const float d = imDepth.at<float>(v,u);

        // 如果获取到的深度点合法(d>0), 那么就保存这个特征点的深度,并且计算出等效的\在假想的右图中该特征点所匹配的特征点的横坐标
        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

/** 
 * @brief 当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
 */
bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

/** 
 * @brief 是否做完预积分
 */
bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

/** 
 * @brief 设置为做完预积分
 */
void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

/** 
 * @brief 左右目模式
 */
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, SPextractor* extractorLeft, SPextractor* extractorRight, SPVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpSPvocabulary(voc),mpExtractorLeft(extractorLeft),mpExtractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info

    // mnScaleLevels = mpORBextractorLeft->GetLevels();
    // mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // mfLogScaleFactor = log(mfScaleFactor);
    // mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpExtractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractKeyPoints,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractKeyPoints,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // 左图中提取的特征点数目
    Nleft = mvKeys.size();
    // 右图中提取的特征点数目
    Nright = mvKeysRight.size();
    // 特征点总数
    N = Nleft + Nright;

    if (RoverZedFrameDebugEnabled())
    {
        std::cout << "[ZED FRAME DEBUG] ctor=two_camera_after_extract"
                  << " ts=" << std::fixed << std::setprecision(9) << mTimeStamp
                  << " N=" << N
                  << " Nleft=" << Nleft
                  << " Nright=" << Nright
                  << " left_desc_rows=" << mDescriptors.rows
                  << " right_desc_rows=" << mDescriptorsRight.rows
                  << std::endl;
    }

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
    ApplyIgevDepthIfAvailable();
    if (RoverZedFrameDebugEnabled())
    {
        std::cout << "[ZED FRAME DEBUG] ctor=two_camera_stereo"
                  << " ts=" << std::fixed << std::setprecision(9) << mTimeStamp
                  << " N=" << N
                  << " Nleft=" << Nleft
                  << " Nright=" << Nright
                  << " stereo_valid=" << CountValidStereoDepth(mvDepth)
                  << std::endl;
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

/** 
 * @brief 左右目匹配
 */
void Frame::ComputeStereoFishEyeMatches()
{
    // 1. 分别取出特征点
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    // 2. 分别取出描述子
    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    // 一些在当前模式用不到的变量给他填一下
    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    // 3. 暴力匹配
    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it)
    {
        // 对于每一对匹配的候选中，最小距离比次小距离的0.7倍还小的认为是好的匹配
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7)
        {
            //For every good match, check parallax and reprojection error to discard spurious matches
            // 对于好的匹配，做三角化，且深度值有效的放入结果
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave],
                  sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            // 三角化
            float depth =
                static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(
                    mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],
                    mvKeysRight[(*it)[0].trainIdx + monoRight],
                    mRlr, mtlr, sigma1, sigma2, p3D);
            // 填充数据
            if(depth > 0.0001f)
            {
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

/** 
 * @brief 两个相机模式下单相机的验证，判断路标点是否在视野中
 */
bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    // Step 1 获得这个地图点的世界坐标
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight)
    {
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else
    {
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    // Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，直接返回false
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight)
        uv = mpCamera2->project(Pc);
    else
        uv = mpCamera->project(Pc);

    // Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
    // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
    // mOw：当前相机光心在世界坐标系下坐标
    const Eigen::Vector3f PO = P - twc;
    // 取模就得到了距离
    const float dist = PO.norm();

    // 如果不在允许的尺度变化范围内，认为重投影不可靠
    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于cos(viewingCosLimit), 即夹角大于viewingCosLimit弧度则返回
    Eigen::Vector3f Pn = pMP->GetNormal();

    // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
    const float viewCos = PO.dot(Pn) / dist;

    // 如果大于给定的阈值 cos(60°)=0.5，认为这个点方向太偏了，重投影不可靠，返回false
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Step 7 记录计算得到的一些参数
    if(bRight)
    {
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else
    {
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

/** 
 * @brief 根据位姿将第i个点投到世界坐标系下
 */
Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM


void ORB_SLAM3::Frame::ApplyIgevDepthIfAvailable()
{
    if (!ORB_SLAM3::RoverIgevDepth::Enabled()) return;
    if (N <= 0) return;

    cv::Mat igevDepth;
    if (!ORB_SLAM3::RoverIgevDepth::LoadDepthForTimestamp(mTimeStamp, igevDepth)) return;
    if (igevDepth.empty()) return;

    const int oldN = N;

    if ((int)mvDepth.size() != oldN)
        mvDepth.resize(oldN, -1.0f);

    if ((int)mvuRight.size() != oldN)
        mvuRight.resize(oldN, -1.0f);

    std::vector<float> mvIgevDepth(oldN, -1.0f);
    std::vector<float> mvDepthConfidence(oldN, 0.0f);
    std::vector<unsigned char> mvDepthSource(oldN, 4);

    const float minDepth = 0.10f;
    const float maxDepth = 60.0f;
    const float highConsistencyRel = 0.25f;
    const float midConsistencyRel  = 0.50f;
    const float edgeRelThreshold   = 0.35f;

    int gridCols = 8;
    int gridRows = 6;
    int finalFeatureLimit = 1200;

    if (const char* env = std::getenv("ROVER_IGEV_GRID_COLS"))
        gridCols = std::max(1, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_GRID_ROWS"))
        gridRows = std::max(1, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_FINAL_FEATURES"))
        finalFeatureLimit = std::max(100, std::atoi(env));

    int maxIgevOnlyPerCell = 8;
    float minIgevOnlyFeatureScore = 0.15f;
    int allowIgevOnly = 1;

    // H1: inverse-depth fusion between stereo and calibrated IGEV.
    // Disabled by default to preserve original G1 behavior.
    int useIgevFuse50 = 0;
    float igevFuseWeight = 0.50f;
    float igevFuseHighWeight = -1.0f;
    float igevFuseMidWeight = -1.0f;
    int useIgevAdaptiveFuse = 0;
    float igevAdaptiveMinWeight = 0.50f;
    float igevAdaptiveMaxWeight = 0.75f;
    int useIgevEdgeAwareFuse = 0;
    float igevFuseHighEdgeWeight = 0.50f;
    float igevFuseMidEdgeWeight = 0.25f;
    int useAdaptiveIgevOnlyQuota = 0;
    int targetDepthPerCell = 8;
    int adaptiveIgevOnlyPerCellMax = maxIgevOnlyPerCell;
    int useConservativeIgevOnlyQuota = 0;
    int conservativeQuotaMin = 6;
    int conservativeQuotaMax = 8;
    int useIgevOnlyRiskGate = 0;
    float igevOnlyMinDisparity = 2.0f;
    float igevOnlyFarDisparity = 5.0f;
    float igevOnlyFarMinScore = 0.20f;
    int useIgevOnlyLocalVarGate = 0;
    float igevOnlyLocalRelMax = 0.20f;
    int useNeedBasedCellRecovery = 0;
    int useOverflowAwareRecovery = 0;
    int recoveryStereoSparseThreshold = 8;
    int recoveryStereoEmptyThreshold = 2;
    int recoveryMinEligibleCandidates = 2;
    int recoveryMinHighScoreCandidates = 1;
    int recoveryOverflowPlusOne = 2;
    int recoveryOverflowFull = 4;
    int recoveryFullHighScoreCandidates = 3;
    float recoveryHighScoreThreshold = 0.25f;
    float recoveryEdgeRatioMax = 0.35f;
    float recoveryConflictRatioMax = 0.40f;
    int useConflictDisparityStats = 0;
    int useDisparityStats = 0;
    int useGeometryConfidenceFusion = 0;
    int useFixedIgevFusionOnly = 0;
    int useIgevSpatialOnly = 0;
    float fixedIgevFusionWeight = 0.50f;
    int useContinuousFrameWeight = 0;
    int useContinuousThreeAnchorDemand = 0;
    int useContinuousConfidenceFusion = 0;
    int useContinuousMidConfidenceOnly = 0;
    float continuousConfidenceTau = 0.15f;
    float continuousStereoHigh = 0.65f;
    float continuousStereoBalanced = 0.50f;
    float continuousStereoLow = 0.35f;
    float continuousEmaAlpha = 0.20f;
    float continuousHighLight = 0.50f;
    float continuousHighBalanced = 0.65f;
    float continuousHighRobust = 0.75f;
    float continuousMidLight = 0.25f;
    float continuousMidBalanced = 0.50f;
    float continuousMidRobust = 0.50f;
    float continuousHighEdgeLight = 0.25f;
    float continuousHighEdgeBalanced = 0.40f;
    float continuousHighEdgeRobust = 0.50f;
    float continuousMidEdgeLight = 0.10f;
    float continuousMidEdgeBalanced = 0.25f;
    float continuousMidEdgeRobust = 0.25f;
    float continuousStereoRatioRaw = 0.0f;
    float continuousStereoRatioEma = 0.0f;
    float continuousSceneDemand = 0.0f;
    float continuousHighWeightCommanded = 0.0f;
    float continuousMidWeightCommanded = 0.0f;
    float continuousHighEdgeWeightCommanded = 0.0f;
    float continuousMidEdgeWeightCommanded = 0.0f;
    int continuousDemandSaturatedLight = 0;
    int continuousDemandSaturatedRobust = 0;
    float continuousConfidenceSum = 0.0f;
    float continuousConfidenceHighSum = 0.0f;
    float continuousConfidenceMidSum = 0.0f;
    float continuousFinalWeightSum = 0.0f;
    int continuousConfidenceCount = 0;
    int continuousConfidenceHighCount = 0;
    int continuousConfidenceMidCount = 0;
    float geometryFusionMinWeight = 0.20f;
    float geometryFusionMaxWeight = 0.85f;
    float geometryStrongStereoDisp = 10.0f;
    float geometryMidStereoDisp = 5.0f;
    float geometryWeakStereoDisp = 2.0f;
    float geometryStrongStereoPenalty = 0.10f;
    float geometryWeakStereoBoost = 0.10f;
    float geometryMidStereoBoost = 0.05f;

    if (const char* env = std::getenv("ROVER_IGEV_FUSE50"))
        useIgevFuse50 = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_FUSE50_WEIGHT"))
        igevFuseWeight = static_cast<float>(std::atof(env));

    igevFuseWeight =
        std::max(0.0f, std::min(1.0f, igevFuseWeight));

    if (const char* env = std::getenv("ROVER_IGEV_FUSE_HIGH_WEIGHT"))
        igevFuseHighWeight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_HIGH_WEIGHT"))
        igevFuseHighWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_FUSE_MID_WEIGHT"))
        igevFuseMidWeight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_MID_WEIGHT"))
        igevFuseMidWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_FUSE_ADAPTIVE"))
        useIgevAdaptiveFuse = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_FUSE_ADAPTIVE_MIN_WEIGHT"))
        igevAdaptiveMinWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_FUSE_ADAPTIVE_MAX_WEIGHT"))
        igevAdaptiveMaxWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_EDGE_AWARE"))
        useIgevEdgeAwareFuse = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_HIGH_EDGE_WEIGHT"))
        igevFuseHighEdgeWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_MID_EDGE_WEIGHT"))
        igevFuseMidEdgeWeight = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_ADAPTIVE_IGEV_ONLY_QUOTA"))
        useAdaptiveIgevOnlyQuota = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_CONSERVATIVE_QUOTA"))
        useConservativeIgevOnlyQuota = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_TARGET_DEPTH_PER_CELL"))
        targetDepthPerCell = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_PER_CELL_MAX"))
        adaptiveIgevOnlyPerCellMax = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_QUOTA_MIN"))
        conservativeQuotaMin = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_QUOTA_MAX"))
        conservativeQuotaMax = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_RISK_GATE"))
        useIgevOnlyRiskGate = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_MIN_DISPARITY"))
        igevOnlyMinDisparity = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_FAR_DISPARITY"))
        igevOnlyFarDisparity = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_FAR_MIN_SCORE"))
        igevOnlyFarMinScore =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_LOCAL_VAR_GATE"))
        useIgevOnlyLocalVarGate = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_LOCAL_REL_MAX"))
        igevOnlyLocalRelMax =
            std::max(0.01f, static_cast<float>(std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_NEED_BASED_CELL_RECOVERY"))
        useNeedBasedCellRecovery = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_OVERFLOW_RECOVERY"))
        useOverflowAwareRecovery = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_STEREO_SPARSE"))
        recoveryStereoSparseThreshold = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_STEREO_EMPTY"))
        recoveryStereoEmptyThreshold = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_MIN_ELIGIBLE"))
        recoveryMinEligibleCandidates = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_MIN_HIGH_SCORE"))
        recoveryMinHighScoreCandidates = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_OVERFLOW_PLUS1"))
        recoveryOverflowPlusOne = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_OVERFLOW_FULL"))
        recoveryOverflowFull = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_FULL_HIGH_SCORE"))
        recoveryFullHighScoreCandidates = std::max(0, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_HIGH_SCORE"))
        recoveryHighScoreThreshold =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_EDGE_RATIO_MAX"))
        recoveryEdgeRatioMax =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_RECOVERY_CONFLICT_RATIO_MAX"))
        recoveryConflictRatioMax =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_CONFLICT_DISP_STATS"))
        useConflictDisparityStats = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_DISPARITY_STATS"))
        useDisparityStats = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_GEOMETRY_CONF_FUSION"))
        useGeometryConfidenceFusion = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_FIXED_FUSION_ONLY"))
        useFixedIgevFusionOnly = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_SPATIAL_ONLY"))
        useIgevSpatialOnly = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_FIXED_FUSION_WEIGHT"))
        fixedIgevFusionWeight = static_cast<float>(std::atof(env));

    fixedIgevFusionWeight =
        std::max(0.0f, std::min(1.0f, fixedIgevFusionWeight));

    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_FRAME_WEIGHT"))
        useContinuousFrameWeight = std::atoi(env);
    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_THREE_ANCHOR_DEMAND"))
        useContinuousThreeAnchorDemand = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_CONFIDENCE_FUSION"))
        useContinuousConfidenceFusion = std::atoi(env);
    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_MID_CONFIDENCE_ONLY"))
        useContinuousMidConfidenceOnly = std::atoi(env);
    if (const char* env = std::getenv("ROVER_IGEV_CONFIDENCE_TAU"))
        continuousConfidenceTau =
            std::max(1e-6f, static_cast<float>(std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_STEREO_HIGH"))
        continuousStereoHigh = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_STEREO_BALANCED"))
        continuousStereoBalanced = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_STEREO_LOW"))
        continuousStereoLow = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONTINUOUS_EMA_ALPHA"))
        continuousEmaAlpha = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_LIGHT"))
        continuousHighLight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_BALANCED"))
        continuousHighBalanced = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_ROBUST"))
        continuousHighRobust = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_LIGHT"))
        continuousMidLight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_BALANCED"))
        continuousMidBalanced = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_ROBUST"))
        continuousMidRobust = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_EDGE_LIGHT"))
        continuousHighEdgeLight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_EDGE_BALANCED"))
        continuousHighEdgeBalanced = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_HIGH_EDGE_ROBUST"))
        continuousHighEdgeRobust = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_EDGE_LIGHT"))
        continuousMidEdgeLight = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_EDGE_BALANCED"))
        continuousMidEdgeBalanced = static_cast<float>(std::atof(env));
    if (const char* env = std::getenv("ROVER_IGEV_CONT_MID_EDGE_ROBUST"))
        continuousMidEdgeRobust = static_cast<float>(std::atof(env));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_MIN_WEIGHT"))
        geometryFusionMinWeight =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_MAX_WEIGHT"))
        geometryFusionMaxWeight =
            std::max(0.0f, std::min(1.0f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_STRONG_STEREO_DISP"))
        geometryStrongStereoDisp =
            std::max(0.01f, static_cast<float>(std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_MID_STEREO_DISP"))
        geometryMidStereoDisp =
            std::max(0.01f, static_cast<float>(std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_WEAK_STEREO_DISP"))
        geometryWeakStereoDisp =
            std::max(0.01f, static_cast<float>(std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_STRONG_STEREO_PENALTY"))
        geometryStrongStereoPenalty =
            std::max(0.0f, std::min(0.5f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_WEAK_STEREO_BOOST"))
        geometryWeakStereoBoost =
            std::max(0.0f, std::min(0.5f, static_cast<float>(std::atof(env))));

    if (const char* env = std::getenv("ROVER_IGEV_GEOM_MID_STEREO_BOOST"))
        geometryMidStereoBoost =
            std::max(0.0f, std::min(0.5f, static_cast<float>(std::atof(env))));

    if (igevFuseHighWeight < 0.0f)
        igevFuseHighWeight = igevFuseWeight;
    if (igevFuseMidWeight < 0.0f)
        igevFuseMidWeight = igevFuseWeight;

    igevFuseHighWeight =
        std::max(0.0f, std::min(1.0f, igevFuseHighWeight));
    igevFuseMidWeight =
        std::max(0.0f, std::min(1.0f, igevFuseMidWeight));
    igevAdaptiveMinWeight =
        std::max(0.0f, std::min(1.0f, igevAdaptiveMinWeight));
    igevAdaptiveMaxWeight =
        std::max(0.0f, std::min(1.0f, igevAdaptiveMaxWeight));
    igevFuseHighEdgeWeight =
        std::max(0.0f, std::min(1.0f, igevFuseHighEdgeWeight));
    igevFuseMidEdgeWeight =
        std::max(0.0f, std::min(1.0f, igevFuseMidEdgeWeight));
    if (igevAdaptiveMinWeight > igevAdaptiveMaxWeight)
        std::swap(igevAdaptiveMinWeight, igevAdaptiveMaxWeight);
    if (geometryFusionMinWeight > geometryFusionMaxWeight)
        std::swap(geometryFusionMinWeight, geometryFusionMaxWeight);
    if (geometryWeakStereoDisp > geometryMidStereoDisp)
        std::swap(geometryWeakStereoDisp, geometryMidStereoDisp);
    if (geometryMidStereoDisp > geometryStrongStereoDisp)
        std::swap(geometryMidStereoDisp, geometryStrongStereoDisp);
    continuousEmaAlpha =
        std::max(0.01f, std::min(1.0f, continuousEmaAlpha));
    continuousStereoHigh =
        std::max(0.0f, std::min(1.0f, continuousStereoHigh));
    continuousStereoBalanced =
        std::max(0.0f, std::min(1.0f, continuousStereoBalanced));
    continuousStereoLow =
        std::max(0.0f, std::min(1.0f, continuousStereoLow));
    if (continuousStereoLow > continuousStereoHigh)
        std::swap(continuousStereoLow, continuousStereoHigh);
    continuousStereoBalanced =
        std::max(
            continuousStereoLow,
            std::min(continuousStereoHigh, continuousStereoBalanced));
    continuousHighLight =
        std::max(0.0f, std::min(1.0f, continuousHighLight));
    continuousHighBalanced =
        std::max(0.0f, std::min(1.0f, continuousHighBalanced));
    continuousHighRobust =
        std::max(0.0f, std::min(1.0f, continuousHighRobust));
    continuousMidLight =
        std::max(0.0f, std::min(1.0f, continuousMidLight));
    continuousMidBalanced =
        std::max(0.0f, std::min(1.0f, continuousMidBalanced));
    continuousMidRobust =
        std::max(0.0f, std::min(1.0f, continuousMidRobust));
    continuousHighEdgeLight =
        std::max(0.0f, std::min(1.0f, continuousHighEdgeLight));
    continuousHighEdgeBalanced =
        std::max(0.0f, std::min(1.0f, continuousHighEdgeBalanced));
    continuousHighEdgeRobust =
        std::max(0.0f, std::min(1.0f, continuousHighEdgeRobust));
    continuousMidEdgeLight =
        std::max(0.0f, std::min(1.0f, continuousMidEdgeLight));
    continuousMidEdgeBalanced =
        std::max(0.0f, std::min(1.0f, continuousMidEdgeBalanced));
    continuousMidEdgeRobust =
        std::max(0.0f, std::min(1.0f, continuousMidEdgeRobust));
    int igevOnlyMonoResidual = 0;

    int igevInvDepthCalib = 0;
    int igevCalibMinPairs = 30;
    float igevCalibMaxRelDiff = 0.30f;
    float igevCalibAlpha = 0.50f;
    float igevCalibScaleMin = 0.90f;
    float igevCalibScaleMax = 1.10f;

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_PER_CELL"))
        maxIgevOnlyPerCell = std::max(0, std::atoi(env));
    adaptiveIgevOnlyPerCellMax =
        std::max(0, std::min(adaptiveIgevOnlyPerCellMax, maxIgevOnlyPerCell));
    conservativeQuotaMin =
        std::max(0, std::min(conservativeQuotaMin, adaptiveIgevOnlyPerCellMax));
    conservativeQuotaMax =
        std::max(0, std::min(conservativeQuotaMax, adaptiveIgevOnlyPerCellMax));
    if (conservativeQuotaMin > conservativeQuotaMax)
        std::swap(conservativeQuotaMin, conservativeQuotaMax);
    igevOnlyMinDisparity = std::max(0.0f, igevOnlyMinDisparity);
    igevOnlyFarDisparity = std::max(igevOnlyMinDisparity, igevOnlyFarDisparity);

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_MIN_SCORE"))
        minIgevOnlyFeatureScore = std::max(0.0f, std::min(1.0f, (float)std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_ALLOW_IGEV_ONLY"))
        allowIgevOnly = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_ONLY_MONO_RESIDUAL"))
        igevOnlyMonoResidual = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_INVDEPTH_CALIB"))
        igevInvDepthCalib = std::atoi(env);

    if (const char* env = std::getenv("ROVER_IGEV_CALIB_MIN_PAIRS"))
        igevCalibMinPairs = std::max(5, std::atoi(env));

    if (const char* env = std::getenv("ROVER_IGEV_CALIB_MAX_REL"))
        igevCalibMaxRelDiff =
            std::max(0.05f, std::min(1.00f, (float)std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_CALIB_ALPHA"))
        igevCalibAlpha =
            std::max(0.0f, std::min(1.0f, (float)std::atof(env)));

    if (const char* env = std::getenv("ROVER_IGEV_CALIB_SCALE_MIN"))
        igevCalibScaleMin = (float)std::atof(env);

    if (const char* env = std::getenv("ROVER_IGEV_CALIB_SCALE_MAX"))
        igevCalibScaleMax = (float)std::atof(env);

    if (igevCalibScaleMin > igevCalibScaleMax)
        std::swap(igevCalibScaleMin, igevCalibScaleMax);

    int igevCalibRawPairs = 0;
    int igevCalibUsedPairs = 0;
    float igevCalibRawScale = 1.0f;
    float igevCalibAppliedScale = 1.0f;
    bool igevCalibApplied = false;
    float igevCalibMedianRelBefore = 0.0f;
    float igevCalibMedianRelAfter = 0.0f;
    float igevCalibP90RelBefore = 0.0f;
    float igevCalibP90RelAfter = 0.0f;
    const char* evidenceCsvPath = std::getenv("ROVER_EVIDENCE_FRAME_CSV");
    const bool evidenceFrameLogEnabled =
        evidenceCsvPath != nullptr && evidenceCsvPath[0] != '\0';
    const char* profileMonitorCsvPath = std::getenv("ROVER_PROFILE_MONITOR_CSV");
    const bool profileMonitorLogEnabled =
        profileMonitorCsvPath != nullptr && profileMonitorCsvPath[0] != '\0';
    const char* evidenceProfileName = std::getenv("ROVER_EVIDENCE_PROFILE_NAME");
    const char* evidenceVariantName = std::getenv("ROVER_EVIDENCE_VARIANT_NAME");

    auto medianValue = [](std::vector<float> values) -> float
    {
        if (values.empty())
            return 1.0f;

        const size_t middle = values.size() / 2;
        std::nth_element(
            values.begin(),
            values.begin() + middle,
            values.end());

        float med = values[middle];

        if ((values.size() % 2) == 0 && middle > 0)
        {
            std::nth_element(
                values.begin(),
                values.begin() + middle - 1,
                values.end());

            med = 0.5f * (med + values[middle - 1]);
        }

        return med;
    };

    auto percentileValue = [](std::vector<float> values, float percentile) -> float
    {
        if (values.empty())
            return 0.0f;

        percentile = std::max(0.0f, std::min(1.0f, percentile));
        const size_t idx = std::min(
            values.size() - 1,
            static_cast<size_t>(std::floor(percentile * (values.size() - 1))));

        std::nth_element(values.begin(), values.begin() + idx, values.end());
        return values[idx];
    };

    if (igevInvDepthCalib &&
        !igevDepth.empty() &&
        oldN > 0 &&
        !mvKeysUn.empty())
    {
        if (igevDepth.type() != CV_32FC1)
        {
            cv::Mat convertedDepth;
            igevDepth.convertTo(convertedDepth, CV_32FC1);
            igevDepth = convertedDepth;
        }

        std::vector<float> inverseDepthRatios;
        inverseDepthRatios.reserve(oldN);

        const int usableN = std::min(
            oldN,
            std::min(
                (int)mvKeysUn.size(),
                std::min((int)mvDepth.size(), (int)mvuRight.size())));

        for (int i = 0; i < usableN; ++i)
        {
            const float zStereo = mvDepth[i];

            if (!std::isfinite(zStereo) ||
                zStereo <= 0.20f ||
                zStereo >= 60.0f ||
                mvuRight[i] < 0.0f)
            {
                continue;
            }

            const cv::KeyPoint& kp = mvKeysUn[i];

            const int u = std::max(
                0,
                std::min(
                    igevDepth.cols - 1,
                    (int)std::lround(kp.pt.x)));

            const int v = std::max(
                0,
                std::min(
                    igevDepth.rows - 1,
                    (int)std::lround(kp.pt.y)));

            const float zIgevRaw = igevDepth.at<float>(v, u);

            if (!std::isfinite(zIgevRaw) ||
                zIgevRaw <= 0.20f ||
                zIgevRaw >= 60.0f)
            {
                continue;
            }

            const float relDiff =
                std::fabs(zIgevRaw - zStereo) /
                std::max(0.10f, zStereo);

            if (relDiff > igevCalibMaxRelDiff)
                continue;

            const float inverseDepthScale = zIgevRaw / zStereo;

            if (!std::isfinite(inverseDepthScale) ||
                inverseDepthScale <= 0.0f)
            {
                continue;
            }

            inverseDepthRatios.push_back(inverseDepthScale);
        }

        igevCalibRawPairs = (int)inverseDepthRatios.size();

        if (igevCalibRawPairs >= igevCalibMinPairs)
        {
            const float firstMedian = medianValue(inverseDepthRatios);

            std::vector<float> deviations;
            deviations.reserve(inverseDepthRatios.size());

            for (float ratio : inverseDepthRatios)
                deviations.push_back(std::fabs(ratio - firstMedian));

            const float mad = medianValue(deviations);

            const float robustThreshold =
                std::max(0.01f, 2.5f * 1.4826f * mad);

            std::vector<float> robustRatios;
            robustRatios.reserve(inverseDepthRatios.size());

            for (float ratio : inverseDepthRatios)
            {
                if (std::fabs(ratio - firstMedian) <= robustThreshold)
                    robustRatios.push_back(ratio);
            }

            igevCalibUsedPairs = (int)robustRatios.size();

            if (igevCalibUsedPairs >= igevCalibMinPairs)
            {
                igevCalibRawScale = medianValue(robustRatios);

                const float clampedScale =
                    std::max(
                        igevCalibScaleMin,
                        std::min(
                            igevCalibScaleMax,
                            igevCalibRawScale));

                igevCalibAppliedScale =
                    1.0f +
                    igevCalibAlpha * (clampedScale - 1.0f);

                if (std::isfinite(igevCalibAppliedScale) &&
                    igevCalibAppliedScale > 0.01f)
                {
                    igevDepth *= (1.0f / igevCalibAppliedScale);
                    igevCalibApplied = true;
                }
            }
        }

        if (evidenceFrameLogEnabled && !inverseDepthRatios.empty())
        {
            std::vector<float> relBefore;
            std::vector<float> relAfter;
            relBefore.reserve(inverseDepthRatios.size());
            relAfter.reserve(inverseDepthRatios.size());

            const float afterScale =
                (std::isfinite(igevCalibAppliedScale) &&
                 igevCalibAppliedScale > 0.01f)
                    ? igevCalibAppliedScale
                    : 1.0f;

            for (float ratio : inverseDepthRatios)
            {
                relBefore.push_back(std::fabs(ratio - 1.0f));
                relAfter.push_back(std::fabs((ratio / afterScale) - 1.0f));
            }

            igevCalibMedianRelBefore = medianValue(relBefore);
            igevCalibMedianRelAfter = medianValue(relAfter);
            igevCalibP90RelBefore = percentileValue(relBefore, 0.90f);
            igevCalibP90RelAfter = percentileValue(relAfter, 0.90f);
        }

        std::cout
            << "[IGEV G1 CALIB]"
            << " frame_ts=" << mTimeStamp
            << " raw_pairs=" << igevCalibRawPairs
            << " used_pairs=" << igevCalibUsedPairs
            << " raw_inv_scale=" << igevCalibRawScale
            << " applied_scale=" << igevCalibAppliedScale
            << " alpha=" << igevCalibAlpha
            << " applied=" << (igevCalibApplied ? 1 : 0);

        if (evidenceFrameLogEnabled)
        {
            std::cout
                << " median_rel_before=" << igevCalibMedianRelBefore
                << " median_rel_after=" << igevCalibMedianRelAfter
                << " p90_rel_before=" << igevCalibP90RelBefore
                << " p90_rel_after=" << igevCalibP90RelAfter;
        }

        std::cout << std::endl;
    }

    const int nCells = gridCols * gridRows;

    int injected = 0;
    int replaced_high = 0;
    int replaced_mid = 0;
    int kept_original = 0;
    int conflict_keep_stereo = 0;
    int rejected = 0;
    int edge_penalty = 0;

    int consistent_count = 0;
    int conflict_count = 0;
    int igev_only_count = 0;
    int stereo_only_count = 0;
    int invalid_count = 0;
    int igev_only_suppressed = 0;
    int igev_only_mono = 0;

    int fuse50_high = 0;
    int fuse50_mid = 0;
    int adaptiveFuseCount = 0;
    int adaptiveEdgeCount = 0;
    float adaptiveWeightSum = 0.0f;
    int geometryFusionCount = 0;
    int geometryFusionHighCount = 0;
    int geometryFusionMidCount = 0;
    int geometryFusionEdgeCount = 0;
    int geometryFusionStrongStereoCount = 0;
    int geometryFusionWeakStereoCount = 0;
    float geometryFusionWeightSum = 0.0f;
    std::vector<float> effectiveIgevWeights;
    effectiveIgevWeights.reserve(oldN);
    float effectiveIgevWeightSum = 0.0f;
    int edgeAwareHighCount = 0;
    int edgeAwareMidCount = 0;
    int adaptiveQuotaAllowedSum = 0;
    int adaptiveQuotaSuppressed = 0;
    int conservativeQuotaMinCount = 0;
    int conservativeQuotaMidCount = 0;
    int conservativeQuotaMaxCount = 0;
    int conservativeQuotaSum = 0;
    int conservativeQuotaDecisions = 0;
    int riskGateRejectDisparity = 0;
    int riskGateRejectScore = 0;
    int riskGateRejectLocalVar = 0;
    int riskGatePass = 0;
    float riskGateLocalRelSum = 0.0f;
    int riskGateLocalRelCount = 0;
    int recoveryCellCount = 0;
    int recoveryPlusOneCount = 0;
    int recoveryFullEightCount = 0;
    int recoveryVetoEdge = 0;
    int recoveryVetoConflict = 0;
    int recoveryStereoSparseCells = 0;
    int recoveryDepthEmptyCells = 0;
    int recoveryEligibleCells = 0;
    int recoveryBaseQuotaSum = 0;
    int recoveryFinalQuotaSum = 0;
    int recoveryOverflowCells = 0;
    int recoveryOverflowCandidates = 0;
    int recoveryNoRoomCells = 0;
    int recoveryHighScoreOverflowCells = 0;
    int disparityBinGe10 = 0;
    int disparityBin5To10 = 0;
    int disparityBin2To5 = 0;
    int disparityBin1To2 = 0;
    int disparityBinLt1 = 0;
    float relDiffSum = 0.0f;
    int relDiffCount = 0;
    int conflictDispGe10 = 0;
    int conflictDisp5To10 = 0;
    int conflictDisp2To5 = 0;
    int conflictDispLt2 = 0;
    int conflictEdgeGe10 = 0;
    int conflictEdge5To10 = 0;
    int conflictEdge2To5 = 0;
    int conflictEdgeLt2 = 0;
    float conflictRelGe10 = 0.0f;
    float conflictRel5To10 = 0.0f;
    float conflictRel2To5 = 0.0f;
    float conflictRelLt2 = 0.0f;
    float conflictScoreGe10 = 0.0f;
    float conflictScore5To10 = 0.0f;
    float conflictScore2To5 = 0.0f;
    float conflictScoreLt2 = 0.0f;
    int stereoValidDepthCount = 0;
    int stereoLowDisparityCount = 0;
    int igevValidDepthCount = 0;
    int bothDepthCount = 0;
    int igevOnlyCandidateCount = 0;
    int igevOnlyEligibleCount = 0;
    float stereoDisparitySum = 0.0f;

    float maxResponse = 0.0f;
    for (int i = 0; i < oldN && i < (int)mvKeysUn.size(); ++i)
        maxResponse = std::max(maxResponse, mvKeysUn[i].response);

    if (maxResponse <= 1e-6f)
        maxResponse = 1.0f;

    auto validDepth = [&](float z) -> bool
    {
        return std::isfinite(z) && z > minDepth && z < maxDepth;
    };

    auto interpolateThreeAnchors = [](float demand,
                                      float lightValue,
                                      float balancedValue,
                                      float robustValue) -> float
    {
        demand = std::max(0.0f, std::min(1.0f, demand));

        if (demand <= 0.5f)
        {
            const float t = demand / 0.5f;
            return lightValue + t * (balancedValue - lightValue);
        }

        const float t = (demand - 0.5f) / 0.5f;
        return balancedValue + t * (robustValue - balancedValue);
    };

    if (useContinuousFrameWeight)
    {
        int rawStereoValidDepthCount = 0;
        const int usableN = std::min(oldN, (int)mvDepth.size());

        for (int i = 0; i < usableN; ++i)
        {
            if (validDepth(mvDepth[i]))
                rawStereoValidDepthCount++;
        }

        const float featureCount =
            oldN > 0 ? static_cast<float>(oldN) : 1.0f;
        continuousStereoRatioRaw =
            static_cast<float>(rawStereoValidDepthCount) / featureCount;

        static bool continuousEmaInitialized = false;
        static float continuousStereoRatioEmaState = 0.0f;

        if (!continuousEmaInitialized)
        {
            continuousStereoRatioEmaState = continuousStereoRatioRaw;
            continuousEmaInitialized = true;
        }
        else
        {
            continuousStereoRatioEmaState =
                continuousEmaAlpha * continuousStereoRatioRaw +
                (1.0f - continuousEmaAlpha) * continuousStereoRatioEmaState;
        }

        continuousStereoRatioEma = continuousStereoRatioEmaState;

        if (useContinuousThreeAnchorDemand)
        {
            if (continuousStereoRatioEma >= continuousStereoHigh)
            {
                continuousSceneDemand = 0.0f;
            }
            else if (continuousStereoRatioEma >= continuousStereoBalanced)
            {
                const float denom = std::max(
                    1e-6f,
                    continuousStereoHigh - continuousStereoBalanced);
                continuousSceneDemand =
                    0.5f *
                    (continuousStereoHigh - continuousStereoRatioEma) /
                    denom;
            }
            else if (continuousStereoRatioEma > continuousStereoLow)
            {
                const float denom = std::max(
                    1e-6f,
                    continuousStereoBalanced - continuousStereoLow);
                continuousSceneDemand =
                    0.5f +
                    0.5f *
                    (continuousStereoBalanced - continuousStereoRatioEma) /
                    denom;
            }
            else
            {
                continuousSceneDemand = 1.0f;
            }
        }
        else
        {
            const float denom =
                std::max(1e-6f, continuousStereoHigh - continuousStereoLow);
            continuousSceneDemand =
                (continuousStereoHigh - continuousStereoRatioEma) / denom;
        }
        continuousSceneDemand =
            std::max(0.0f, std::min(1.0f, continuousSceneDemand));

        continuousDemandSaturatedLight =
            continuousSceneDemand <= 1e-6f ? 1 : 0;
        continuousDemandSaturatedRobust =
            continuousSceneDemand >= 1.0f - 1e-6f ? 1 : 0;

        continuousHighWeightCommanded =
            interpolateThreeAnchors(
                continuousSceneDemand,
                continuousHighLight,
                continuousHighBalanced,
                continuousHighRobust);
        continuousMidWeightCommanded =
            interpolateThreeAnchors(
                continuousSceneDemand,
                continuousMidLight,
                continuousMidBalanced,
                continuousMidRobust);
        continuousHighEdgeWeightCommanded =
            interpolateThreeAnchors(
                continuousSceneDemand,
                continuousHighEdgeLight,
                continuousHighEdgeBalanced,
                continuousHighEdgeRobust);
        continuousMidEdgeWeightCommanded =
            interpolateThreeAnchors(
                continuousSceneDemand,
                continuousMidEdgeLight,
                continuousMidEdgeBalanced,
                continuousMidEdgeRobust);

        igevFuseHighWeight = continuousHighWeightCommanded;
        igevFuseMidWeight = continuousMidWeightCommanded;
        igevFuseHighEdgeWeight = continuousHighEdgeWeightCommanded;
        igevFuseMidEdgeWeight = continuousMidEdgeWeightCommanded;
    }
    else
    {
        continuousStereoRatioRaw = 0.0f;
        continuousStereoRatioEma = 0.0f;
        continuousSceneDemand = 0.0f;
        continuousHighWeightCommanded = igevFuseHighWeight;
        continuousMidWeightCommanded = igevFuseMidWeight;
        continuousHighEdgeWeightCommanded = igevFuseHighEdgeWeight;
        continuousMidEdgeWeightCommanded = igevFuseMidEdgeWeight;
    };

    auto isDepthEdge = [&](int u, int v, float z) -> bool
    {
        if (u <= 0 || u >= igevDepth.cols - 1 || v <= 0 || v >= igevDepth.rows - 1)
            return false;

        const float zl = igevDepth.at<float>(v, u - 1);
        const float zr = igevDepth.at<float>(v, u + 1);
        const float zu = igevDepth.at<float>(v - 1, u);
        const float zd = igevDepth.at<float>(v + 1, u);

        float maxRel = 0.0f;

        if (validDepth(zl)) maxRel = std::max(maxRel, std::fabs(zl - z) / std::max(z, minDepth));
        if (validDepth(zr)) maxRel = std::max(maxRel, std::fabs(zr - z) / std::max(z, minDepth));
        if (validDepth(zu)) maxRel = std::max(maxRel, std::fabs(zu - z) / std::max(z, minDepth));
        if (validDepth(zd)) maxRel = std::max(maxRel, std::fabs(zd - z) / std::max(z, minDepth));

        return maxRel > edgeRelThreshold;
    };

    auto localDepthRelRange = [&](int u, int v, float z) -> float
    {
        if (u <= 0 || u >= igevDepth.cols - 1 || v <= 0 || v >= igevDepth.rows - 1)
            return 0.0f;

        float minLocal = z;
        float maxLocal = z;
        int validCount = 0;

        for (int yy = v - 1; yy <= v + 1; ++yy)
        {
            for (int xx = u - 1; xx <= u + 1; ++xx)
            {
                const float zn = igevDepth.at<float>(yy, xx);
                if (!validDepth(zn))
                    continue;

                minLocal = std::min(minLocal, zn);
                maxLocal = std::max(maxLocal, zn);
                validCount++;
            }
        }

        if (validCount < 5)
            return std::numeric_limits<float>::infinity();

        return (maxLocal - minLocal) / std::max(z, minDepth);
    };

    auto geometryFuseWeight = [&](float baseWeight,
                                  float edgeWeight,
                                  float zStereo,
                                  float relDiff,
                                  bool onDepthEdge,
                                  bool calibApplied,
                                  float featureScore) -> float
    {
        float weight = baseWeight;
        const float stereoDisparity =
            zStereo > minDepth ? mbf / zStereo : 0.0f;

        if (stereoDisparity >= geometryStrongStereoDisp)
        {
            weight -= geometryStrongStereoPenalty;
        }
        else if (stereoDisparity < geometryWeakStereoDisp)
        {
            weight += geometryWeakStereoBoost;
        }
        else if (stereoDisparity < geometryMidStereoDisp)
        {
            weight += geometryMidStereoBoost;
        }

        const float consistencyQuality =
            std::max(
                0.0f,
                std::min(
                    1.0f,
                    1.0f - relDiff / std::max(midConsistencyRel, 1e-6f)));

        // Keep the adjustment small: this is confidence shaping, not a new
        // dataset-specific policy.
        weight += 0.05f * (consistencyQuality - 0.5f);

        if (!calibApplied)
            weight = std::min(weight, 0.50f);

        if (onDepthEdge)
            weight = std::min(weight, edgeWeight);

        if (featureScore < 0.10f)
            weight -= 0.05f;
        else if (featureScore > 0.30f)
            weight += 0.03f;

        return std::max(
            geometryFusionMinWeight,
            std::min(geometryFusionMaxWeight, weight));
    };

    struct Candidate
    {
        int idx;
        int cell;
        float score;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(oldN);

    std::vector<int> gridCount(nCells, 0);
    std::vector<int> gridHighConf(nCells, 0);
    std::vector<int> gridConflict(nCells, 0);
    std::vector<int> gridIgevOnlyInjected(nCells, 0);
    std::vector<int> gridStereoDepthCount(nCells, 0);
    std::vector<float> gridConfSum(nCells, 0.0f);
    std::vector<int> cellQuota(nCells, maxIgevOnlyPerCell);
    std::vector<int> cellBaseQuota(nCells, maxIgevOnlyPerCell);
    std::vector<int> cellIgevOnlyEligible(nCells, 0);
    std::vector<int> cellIgevOnlyHighScore(nCells, 0);
    std::vector<int> cellEdgeEvents(nCells, 0);
    std::vector<int> cellConflictEvents(nCells, 0);
    std::vector<int> cellBothDepthEvents(nCells, 0);

    auto h4BaseQuotaForCell = [&](int cell) -> int
    {
        if (!useAdaptiveIgevOnlyQuota || cell < 0 || cell >= nCells)
            return maxIgevOnlyPerCell;

        if (!useConservativeIgevOnlyQuota)
        {
            return std::max(
                0,
                std::min(
                    adaptiveIgevOnlyPerCellMax,
                    targetDepthPerCell - gridStereoDepthCount[cell]));
        }

        const int stereoDepthCount = gridStereoDepthCount[cell];
        int quota = conservativeQuotaMax;

        if (stereoDepthCount >= 2 * targetDepthPerCell)
            quota = conservativeQuotaMin;
        else if (stereoDepthCount >= targetDepthPerCell)
            quota = std::min(conservativeQuotaMax, conservativeQuotaMin + 1);

        return std::max(0, std::min(adaptiveIgevOnlyPerCellMax, quota));
    };

    if (useAdaptiveIgevOnlyQuota)
    {
        for (int i = 0; i < oldN && i < (int)mvKeysUn.size(); ++i)
        {
            const cv::KeyPoint& kp = mvKeysUn[i];
            const int gx = std::min(
                gridCols - 1,
                std::max(0, (int)(kp.pt.x * gridCols / std::max(1, igevDepth.cols))));
            const int gy = std::min(
                gridRows - 1,
                std::max(0, (int)(kp.pt.y * gridRows / std::max(1, igevDepth.rows))));
            const int cell = gy * gridCols + gx;

            if (cell >= 0 && cell < nCells)
            {
                const float zOld = mvDepth[i];
                const bool zOldOk = validDepth(zOld);

                if (zOldOk)
                    gridStereoDepthCount[cell]++;

                if (useNeedBasedCellRecovery)
                {
                    const int u = cvRound(kp.pt.x);
                    const int v = cvRound(kp.pt.y);
                    float zIgev = -1.0f;
                    bool zIgevOk = false;
                    bool onDepthEdge = false;

                    if (u >= 0 && u < igevDepth.cols && v >= 0 && v < igevDepth.rows)
                    {
                        zIgev = igevDepth.at<float>(v, u);
                        zIgevOk = validDepth(zIgev);
                        if (zIgevOk)
                            onDepthEdge = isDepthEdge(u, v, zIgev);
                    }

                    const float featureScorePre =
                        std::min(1.0f, std::max(0.0f, kp.response / maxResponse));

                    if (zIgevOk && !zOldOk)
                    {
                        if (onDepthEdge)
                            cellEdgeEvents[cell]++;

                        if (!onDepthEdge &&
                            featureScorePre >= minIgevOnlyFeatureScore)
                        {
                            cellIgevOnlyEligible[cell]++;
                            if (featureScorePre >= recoveryHighScoreThreshold)
                                cellIgevOnlyHighScore[cell]++;
                        }
                    }
                    else if (zIgevOk && zOldOk)
                    {
                        cellBothDepthEvents[cell]++;
                        if (onDepthEdge)
                            cellEdgeEvents[cell]++;

                        const float relDiff =
                            std::fabs(zOld - zIgev) / std::max(zIgev, minDepth);
                        if (relDiff > midConsistencyRel)
                            cellConflictEvents[cell]++;
                    }
                }
            }
        }

        for (int c = 0; c < nCells; ++c)
        {
            const int baseQuota = h4BaseQuotaForCell(c);
            int finalQuota = baseQuota;

            if (useNeedBasedCellRecovery)
            {
                const bool sparse =
                    gridStereoDepthCount[c] < recoveryStereoSparseThreshold;
                const bool empty =
                    gridStereoDepthCount[c] <= recoveryStereoEmptyThreshold;
                const bool enoughCandidates =
                    cellIgevOnlyEligible[c] >= recoveryMinEligibleCandidates &&
                    cellIgevOnlyHighScore[c] >= recoveryMinHighScoreCandidates;

                const int edgeDenom =
                    std::max(
                        1,
                        cellIgevOnlyEligible[c] +
                            cellEdgeEvents[c] +
                            cellBothDepthEvents[c]);
                const int conflictDenom = std::max(1, cellBothDepthEvents[c]);
                const float edgeRatio =
                    static_cast<float>(cellEdgeEvents[c]) / edgeDenom;
                const float conflictRatio =
                    static_cast<float>(cellConflictEvents[c]) / conflictDenom;
                const bool edgeOk = edgeRatio <= recoveryEdgeRatioMax;
                const bool conflictOk = conflictRatio <= recoveryConflictRatioMax;

                if (sparse)
                    recoveryStereoSparseCells++;
                if (empty)
                    recoveryDepthEmptyCells++;
                if (cellIgevOnlyEligible[c] > 0)
                    recoveryEligibleCells++;

                if (useOverflowAwareRecovery)
                {
                    const int overflow =
                        std::max(cellIgevOnlyEligible[c] - baseQuota, 0);

                    if (overflow > 0)
                    {
                        recoveryOverflowCells++;
                        recoveryOverflowCandidates += overflow;
                    }

                    if (baseQuota >= adaptiveIgevOnlyPerCellMax && overflow > 0)
                        recoveryNoRoomCells++;

                    if (baseQuota < adaptiveIgevOnlyPerCellMax &&
                        overflow >= recoveryOverflowPlusOne &&
                        cellIgevOnlyHighScore[c] >= recoveryMinHighScoreCandidates &&
                        edgeOk &&
                        conflictOk)
                    {
                        if (overflow >= recoveryOverflowFull &&
                            sparse &&
                            cellIgevOnlyHighScore[c] >= recoveryFullHighScoreCandidates)
                        {
                            finalQuota = adaptiveIgevOnlyPerCellMax;
                            recoveryFullEightCount++;
                        }
                        else
                        {
                            finalQuota =
                                std::min(adaptiveIgevOnlyPerCellMax, baseQuota + 1);
                            recoveryPlusOneCount++;
                        }

                        if (cellIgevOnlyHighScore[c] > baseQuota)
                            recoveryHighScoreOverflowCells++;

                        if (finalQuota > baseQuota)
                            recoveryCellCount++;
                    }
                    else if (baseQuota < adaptiveIgevOnlyPerCellMax &&
                             overflow >= recoveryOverflowPlusOne &&
                             cellIgevOnlyHighScore[c] >= recoveryMinHighScoreCandidates)
                    {
                        if (!edgeOk)
                            recoveryVetoEdge++;
                        if (!conflictOk)
                            recoveryVetoConflict++;
                    }
                }
                else if ((sparse || empty) && enoughCandidates && edgeOk && conflictOk)
                {
                    if (empty)
                    {
                        finalQuota = adaptiveIgevOnlyPerCellMax;
                        recoveryFullEightCount++;
                    }
                    else
                    {
                        finalQuota =
                            std::min(adaptiveIgevOnlyPerCellMax, baseQuota + 1);
                        recoveryPlusOneCount++;
                    }

                    if (finalQuota > baseQuota)
                        recoveryCellCount++;
                }
                else if ((sparse || empty) && enoughCandidates)
                {
                    if (!edgeOk)
                        recoveryVetoEdge++;
                    if (!conflictOk)
                        recoveryVetoConflict++;
                }
            }

            cellBaseQuota[c] = baseQuota;
            cellQuota[c] = finalQuota;
            recoveryBaseQuotaSum += baseQuota;
            recoveryFinalQuotaSum += finalQuota;
        }
    }

    float confSum = 0.0f;
    int confCount = 0;

    for (int i = 0; i < oldN; ++i)
    {
        if (i >= (int)mvKeysUn.size())
        {
            rejected++;
            invalid_count++;
            continue;
        }

        const cv::KeyPoint& kp = mvKeysUn[i];

        const int u = cvRound(kp.pt.x);
        const int v = cvRound(kp.pt.y);

        const float zOld = mvDepth[i];
        const bool zOldOk = validDepth(zOld);
        if (zOldOk)
        {
            stereoValidDepthCount++;
            const float disparity = mbf / zOld;
            stereoDisparitySum += disparity;
            if (disparity < 2.0f)
                stereoLowDisparityCount++;
        }
        if (useDisparityStats && zOldOk)
        {
            const float disparity = mbf / zOld;
            if (disparity >= 10.0f) disparityBinGe10++;
            else if (disparity >= 5.0f) disparityBin5To10++;
            else if (disparity >= 2.0f) disparityBin2To5++;
            else if (disparity >= 1.0f) disparityBin1To2++;
            else disparityBinLt1++;
        }

        int gx = std::min(gridCols - 1, std::max(0, (int)(kp.pt.x * gridCols / std::max(1, igevDepth.cols))));
        int gy = std::min(gridRows - 1, std::max(0, (int)(kp.pt.y * gridRows / std::max(1, igevDepth.rows))));
        int cell = gy * gridCols + gx;

        const float featureScorePre = std::min(1.0f, std::max(0.0f, kp.response / maxResponse));

        float zIgev = -1.0f;
        bool zIgevOk = false;
        bool onDepthEdge = false;

        if (u >= 0 && u < igevDepth.cols && v >= 0 && v < igevDepth.rows)
        {
            zIgev = igevDepth.at<float>(v, u);
            zIgevOk = validDepth(zIgev);
            if (zIgevOk)
                onDepthEdge = isDepthEdge(u, v, zIgev);
        }

        mvIgevDepth[i] = zIgev;
        if (zIgevOk)
            igevValidDepthCount++;
        if (zIgevOk && zOldOk)
            bothDepthCount++;
        if (zIgevOk && !zOldOk)
        {
            igevOnlyCandidateCount++;
            if (!onDepthEdge && featureScorePre >= minIgevOnlyFeatureScore)
                igevOnlyEligibleCount++;
        }

        if (zIgevOk && !zOldOk)
        {
            int cellIgevOnlyLimit = maxIgevOnlyPerCell;
            if (useAdaptiveIgevOnlyQuota && cell >= 0 && cell < nCells)
            {
                if (useNeedBasedCellRecovery)
                {
                    const int baseQuota = cellBaseQuota[cell];
                    cellIgevOnlyLimit = cellQuota[cell];

                    if (baseQuota == conservativeQuotaMin)
                        conservativeQuotaMinCount++;
                    else if (baseQuota == std::min(conservativeQuotaMax, conservativeQuotaMin + 1))
                        conservativeQuotaMidCount++;
                    else
                        conservativeQuotaMaxCount++;

                    conservativeQuotaSum += baseQuota;
                    conservativeQuotaDecisions++;
                }
                else if (useConservativeIgevOnlyQuota)
                {
                    const int stereoDepthCount = gridStereoDepthCount[cell];

                    if (stereoDepthCount >= 2 * targetDepthPerCell)
                    {
                        cellIgevOnlyLimit = conservativeQuotaMin;
                        conservativeQuotaMinCount++;
                    }
                    else if (stereoDepthCount >= targetDepthPerCell)
                    {
                        cellIgevOnlyLimit =
                            std::min(conservativeQuotaMax, conservativeQuotaMin + 1);
                        conservativeQuotaMidCount++;
                    }
                    else
                    {
                        cellIgevOnlyLimit = conservativeQuotaMax;
                        conservativeQuotaMaxCount++;
                    }

                    cellIgevOnlyLimit =
                        std::max(
                            0,
                            std::min(adaptiveIgevOnlyPerCellMax, cellIgevOnlyLimit));
                    conservativeQuotaSum += cellIgevOnlyLimit;
                    conservativeQuotaDecisions++;
                }
                else
                {
                    cellIgevOnlyLimit =
                        std::max(
                            0,
                            std::min(
                                adaptiveIgevOnlyPerCellMax,
                                targetDepthPerCell - gridStereoDepthCount[cell]));
                }

                adaptiveQuotaAllowedSum += cellIgevOnlyLimit;
            }

            bool riskGateOk = true;
            const float igevDisparity =
                zIgev > minDepth ? mbf / zIgev : 0.0f;
            float localRelRange = 0.0f;

            if (useIgevOnlyRiskGate)
            {
                if (igevDisparity < igevOnlyMinDisparity)
                {
                    riskGateOk = false;
                    riskGateRejectDisparity++;
                }

                if (riskGateOk &&
                    igevDisparity < igevOnlyFarDisparity &&
                    featureScorePre < igevOnlyFarMinScore)
                {
                    riskGateOk = false;
                    riskGateRejectScore++;
                }

                if (riskGateOk && useIgevOnlyLocalVarGate)
                {
                    localRelRange = localDepthRelRange(u, v, zIgev);
                    riskGateLocalRelSum += localRelRange;
                    riskGateLocalRelCount++;

                    if (!std::isfinite(localRelRange) ||
                        localRelRange > igevOnlyLocalRelMax)
                    {
                        riskGateOk = false;
                        riskGateRejectLocalVar++;
                    }
                }

                if (riskGateOk)
                    riskGatePass++;
            }

            const bool allowThisIgevOnly =
                allowIgevOnly &&
                riskGateOk &&
                !onDepthEdge &&
                featureScorePre >= minIgevOnlyFeatureScore &&
                cell >= 0 &&
                cell < nCells &&
                gridIgevOnlyInjected[cell] < cellIgevOnlyLimit;

            if (allowThisIgevOnly)
            {
                mvDepth[i] = zIgev;

                if (igevOnlyMonoResidual)
                {
                    mvuRight[i] = -1.0f;
                    igev_only_mono++;
                }
                else
                {
                    mvuRight[i] = kp.pt.x - mbf / zIgev;
                }

                mvDepthConfidence[i] = 0.70f;
                mvDepthSource[i] = 1;

                injected++;
                igev_only_count++;
                gridIgevOnlyInjected[cell]++;
            }
            else
            {
                mvDepth[i] = -1.0f;
                mvuRight[i] = -1.0f;

                mvDepthConfidence[i] = onDepthEdge ? 0.10f : 0.25f;
                mvDepthSource[i] = 6;

                rejected++;
                invalid_count++;
                igev_only_suppressed++;
                if (useAdaptiveIgevOnlyQuota &&
                    cell >= 0 &&
                    cell < nCells &&
                    cellIgevOnlyLimit <= gridIgevOnlyInjected[cell])
                {
                    adaptiveQuotaSuppressed++;
                }
                if (onDepthEdge) edge_penalty++;
            }
        }
        else if (zIgevOk && zOldOk)
        {
            if (useIgevSpatialOnly)
            {
                mvDepth[i] = zOld;
                mvuRight[i] = kp.pt.x - mbf / zOld;

                mvDepthConfidence[i] = 0.50f;
                mvDepthSource[i] = 0;

                kept_original++;
                stereo_only_count++;
                continue;
            }

            const float relDiff = std::fabs(zOld - zIgev) / std::max(zIgev, minDepth);
            if (useDisparityStats)
            {
                relDiffSum += relDiff;
                relDiffCount++;
            }
            const float consistencyConfidence =
                std::max(
                    0.0f,
                    std::min(
                        1.0f,
                        1.0f - relDiff / midConsistencyRel));
            const float calibrationConfidence =
                igevCalibApplied
                    ? std::max(
                          0.0f,
                          std::min(
                              1.0f,
                              static_cast<float>(igevCalibUsedPairs) / 100.0f))
                    : 0.0f;
            const float edgeConfidence = onDepthEdge ? 0.25f : 1.0f;
            const float adaptiveConfidence =
                consistencyConfidence *
                calibrationConfidence *
                edgeConfidence;
            const float adaptiveFuseWeight =
                igevAdaptiveMinWeight +
                (igevAdaptiveMaxWeight - igevAdaptiveMinWeight) *
                    adaptiveConfidence;

            if (useFixedIgevFusionOnly)
            {
                const float rhoStereo = 1.0f / zOld;
                const float rhoIgev = 1.0f / zIgev;
                const float rhoFused =
                    (1.0f - fixedIgevFusionWeight) * rhoStereo +
                    fixedIgevFusionWeight * rhoIgev;
                const float zSelected =
                    (std::isfinite(rhoFused) && rhoFused > 1e-6f)
                        ? 1.0f / rhoFused
                        : zOld;

                mvDepth[i] = zSelected;
                mvuRight[i] = kp.pt.x - mbf / zSelected;

                mvDepthConfidence[i] = 0.75f;
                mvDepthSource[i] = 5;
                effectiveIgevWeights.push_back(fixedIgevFusionWeight);
                effectiveIgevWeightSum += fixedIgevFusionWeight;

                replaced_mid++;
                consistent_count++;
            }
            else if (relDiff <= highConsistencyRel)
            {
                float zSelected = zIgev;
                float actualIgevWeight = 1.0f;

                if (useIgevFuse50)
                {
                    const float rhoStereo = 1.0f / zOld;
                    const float rhoIgev = 1.0f / zIgev;
                    float fuseWeight =
                        useGeometryConfidenceFusion
                            ? geometryFuseWeight(
                                  igevFuseHighWeight,
                                  igevFuseHighEdgeWeight,
                                  zOld,
                                  relDiff,
                                  onDepthEdge,
                                  igevCalibApplied,
                                  featureScorePre)
                            : useIgevAdaptiveFuse
                            ? adaptiveFuseWeight
                            : (useIgevEdgeAwareFuse && onDepthEdge
	                                   ? igevFuseHighEdgeWeight
	                                   : igevFuseHighWeight);
                    if (useContinuousConfidenceFusion &&
                        !useContinuousMidConfidenceOnly)
                    {
                        const float confidence =
                            std::exp(-relDiff / continuousConfidenceTau);
                        fuseWeight =
                            std::max(0.0f, std::min(1.0f, fuseWeight * confidence));
                        continuousConfidenceSum += confidence;
                        continuousConfidenceHighSum += confidence;
                        continuousFinalWeightSum += fuseWeight;
                        continuousConfidenceCount++;
                        continuousConfidenceHighCount++;
                    }
                    actualIgevWeight = fuseWeight;

                    const float rhoFused =
                        (1.0f - fuseWeight) * rhoStereo +
                        fuseWeight * rhoIgev;

                    if (std::isfinite(rhoFused) && rhoFused > 1e-6f)
                    {
                        zSelected = 1.0f / rhoFused;
                        fuse50_high++;
                        if (useIgevEdgeAwareFuse && onDepthEdge)
                            edgeAwareHighCount++;
                        if (useIgevAdaptiveFuse)
                        {
                            adaptiveFuseCount++;
                            adaptiveWeightSum += fuseWeight;
                            if (onDepthEdge)
                                adaptiveEdgeCount++;
                        }
                        if (useGeometryConfidenceFusion)
                        {
                            geometryFusionCount++;
                            geometryFusionHighCount++;
                            geometryFusionWeightSum += fuseWeight;
                            if (onDepthEdge)
                                geometryFusionEdgeCount++;
                            const float stereoDisparity =
                                zOld > minDepth ? mbf / zOld : 0.0f;
                            if (stereoDisparity >= geometryStrongStereoDisp)
                                geometryFusionStrongStereoCount++;
                            if (stereoDisparity < geometryWeakStereoDisp)
                                geometryFusionWeakStereoCount++;
                        }
                    }
                }

                mvDepth[i] = zSelected;
                mvuRight[i] = kp.pt.x - mbf / zSelected;

                mvDepthConfidence[i] = onDepthEdge ? 0.65f : 1.00f;
                mvDepthSource[i] = 2;
                effectiveIgevWeights.push_back(actualIgevWeight);
                effectiveIgevWeightSum += actualIgevWeight;

                replaced_high++;
                consistent_count++;
                if (onDepthEdge) edge_penalty++;
            }
            else if (relDiff <= midConsistencyRel)
            {
                float zSelected = zIgev;
                float actualIgevWeight = 1.0f;

                if (useIgevFuse50)
                {
                    const float rhoStereo = 1.0f / zOld;
                    const float rhoIgev = 1.0f / zIgev;
                    float fuseWeight =
                        useGeometryConfidenceFusion
                            ? geometryFuseWeight(
                                  igevFuseMidWeight,
                                  igevFuseMidEdgeWeight,
                                  zOld,
                                  relDiff,
                                  onDepthEdge,
                                  igevCalibApplied,
                                  featureScorePre)
                            : useIgevAdaptiveFuse
                            ? adaptiveFuseWeight
                            : (useIgevEdgeAwareFuse && onDepthEdge
	                                   ? igevFuseMidEdgeWeight
	                                   : igevFuseMidWeight);
                    if (useContinuousConfidenceFusion)
                    {
                        const float confidence =
                            std::exp(-relDiff / continuousConfidenceTau);
                        fuseWeight =
                            std::max(0.0f, std::min(1.0f, fuseWeight * confidence));
                        continuousConfidenceSum += confidence;
                        continuousConfidenceMidSum += confidence;
                        continuousFinalWeightSum += fuseWeight;
                        continuousConfidenceCount++;
                        continuousConfidenceMidCount++;
                    }
                    actualIgevWeight = fuseWeight;

                    const float rhoFused =
                        (1.0f - fuseWeight) * rhoStereo +
                        fuseWeight * rhoIgev;

                    if (std::isfinite(rhoFused) && rhoFused > 1e-6f)
                    {
                        zSelected = 1.0f / rhoFused;
                        fuse50_mid++;
                        if (useIgevEdgeAwareFuse && onDepthEdge)
                            edgeAwareMidCount++;
                        if (useIgevAdaptiveFuse)
                        {
                            adaptiveFuseCount++;
                            adaptiveWeightSum += fuseWeight;
                            if (onDepthEdge)
                                adaptiveEdgeCount++;
                        }
                        if (useGeometryConfidenceFusion)
                        {
                            geometryFusionCount++;
                            geometryFusionMidCount++;
                            geometryFusionWeightSum += fuseWeight;
                            if (onDepthEdge)
                                geometryFusionEdgeCount++;
                            const float stereoDisparity =
                                zOld > minDepth ? mbf / zOld : 0.0f;
                            if (stereoDisparity >= geometryStrongStereoDisp)
                                geometryFusionStrongStereoCount++;
                            if (stereoDisparity < geometryWeakStereoDisp)
                                geometryFusionWeakStereoCount++;
                        }
                    }
                }

                mvDepth[i] = zSelected;
                mvuRight[i] = kp.pt.x - mbf / zSelected;

                mvDepthConfidence[i] = onDepthEdge ? 0.45f : 0.75f;
                mvDepthSource[i] = 5;
                effectiveIgevWeights.push_back(actualIgevWeight);
                effectiveIgevWeightSum += actualIgevWeight;

                replaced_mid++;
                consistent_count++;
                if (onDepthEdge) edge_penalty++;
            }
            else
            {
                if (useConflictDisparityStats)
                {
                    const float stereoDisparity =
                        zOld > minDepth ? mbf / zOld : 0.0f;

                    if (stereoDisparity >= 10.0f)
                    {
                        conflictDispGe10++;
                        conflictRelGe10 += relDiff;
                        conflictScoreGe10 += featureScorePre;
                        if (onDepthEdge)
                            conflictEdgeGe10++;
                    }
                    else if (stereoDisparity >= 5.0f)
                    {
                        conflictDisp5To10++;
                        conflictRel5To10 += relDiff;
                        conflictScore5To10 += featureScorePre;
                        if (onDepthEdge)
                            conflictEdge5To10++;
                    }
                    else if (stereoDisparity >= 2.0f)
                    {
                        conflictDisp2To5++;
                        conflictRel2To5 += relDiff;
                        conflictScore2To5 += featureScorePre;
                        if (onDepthEdge)
                            conflictEdge2To5++;
                    }
                    else
                    {
                        conflictDispLt2++;
                        conflictRelLt2 += relDiff;
                        conflictScoreLt2 += featureScorePre;
                        if (onDepthEdge)
                            conflictEdgeLt2++;
                    }
                }

                mvDepth[i] = zOld;
                mvuRight[i] = kp.pt.x - mbf / zOld;

                mvDepthConfidence[i] = onDepthEdge ? 0.15f : 0.25f;
                mvDepthSource[i] = 3;
                effectiveIgevWeights.push_back(0.0f);

                conflict_keep_stereo++;
                conflict_count++;
                if (onDepthEdge) edge_penalty++;
            }
        }
        else if (!zIgevOk && zOldOk)
        {
            mvDepth[i] = zOld;
            mvuRight[i] = kp.pt.x - mbf / zOld;

            mvDepthConfidence[i] = 0.50f;
            mvDepthSource[i] = 0;

            kept_original++;
            stereo_only_count++;
        }
        else
        {
            mvDepthConfidence[i] = 0.0f;
            mvDepthSource[i] = 4;

            rejected++;
            invalid_count++;
        }

        const float featureScore = std::min(1.0f, std::max(0.0f, kp.response / maxResponse));
        const float depthScore = std::min(1.0f, std::max(0.0f, mvDepthConfidence[i]));

        float sourceBonus = 0.0f;
        if (mvDepthSource[i] == 2) sourceBonus = 0.15f;
        else if (mvDepthSource[i] == 5) sourceBonus = 0.10f;
        else if (mvDepthSource[i] == 1) sourceBonus = 0.05f;
        else if (mvDepthSource[i] == 3) sourceBonus = -0.20f;

        float finalScore = 0.55f * featureScore + 0.40f * depthScore + sourceBonus;

        if (onDepthEdge)
            finalScore *= 0.80f;

        finalScore = std::min(1.0f, std::max(0.0f, finalScore));

        candidates.push_back({i, cell, finalScore});

        gridCount[cell]++;
        gridConfSum[cell] += mvDepthConfidence[i];
        if (mvDepthConfidence[i] >= 0.70f) gridHighConf[cell]++;
        if (mvDepthSource[i] == 3) gridConflict[cell]++;

        confSum += mvDepthConfidence[i];
        confCount++;
    }

    int selectedN = oldN;
    bool gridSelectionEnabled = false;

    if (const char* env = std::getenv("ROVER_IGEV_GRID_SELECT"))
    {
        std::string v(env);
        gridSelectionEnabled = !(v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF");
    }

    if (gridSelectionEnabled && oldN > finalFeatureLimit && finalFeatureLimit > 0)
    {
        std::vector<std::vector<Candidate>> byCell(nCells);
        for (const Candidate& c : candidates)
        {
            if (c.cell >= 0 && c.cell < nCells)
                byCell[c.cell].push_back(c);
        }

        for (int c = 0; c < nCells; ++c)
        {
            std::sort(byCell[c].begin(), byCell[c].end(),
                      [](const Candidate& a, const Candidate& b)
                      {
                          return a.score > b.score;
                      });
        }

        std::vector<unsigned char> chosen(oldN, 0);
        std::vector<int> selected;
        selected.reserve(finalFeatureLimit);

        const int baseQuota = std::max(1, finalFeatureLimit / std::max(1, nCells));

        for (int c = 0; c < nCells; ++c)
        {
            int take = std::min(baseQuota, (int)byCell[c].size());
            for (int k = 0; k < take; ++k)
            {
                const int idx = byCell[c][k].idx;
                if (idx >= 0 && idx < oldN && !chosen[idx])
                {
                    chosen[idx] = 1;
                    selected.push_back(idx);
                }
            }
        }

        std::vector<Candidate> global = candidates;
        std::sort(global.begin(), global.end(),
                  [](const Candidate& a, const Candidate& b)
                  {
                      return a.score > b.score;
                  });

        for (const Candidate& c : global)
        {
            if ((int)selected.size() >= finalFeatureLimit)
                break;

            const int idx = c.idx;
            if (idx >= 0 && idx < oldN && !chosen[idx])
            {
                chosen[idx] = 1;
                selected.push_back(idx);
            }
        }

        std::sort(selected.begin(), selected.end());

        std::vector<cv::KeyPoint> newKeys;
        std::vector<cv::KeyPoint> newKeysUn;
        std::vector<float> newDepth;
        std::vector<float> newRight;
        std::vector<float> newIgevDepth;
        std::vector<float> newDepthConf;
        std::vector<unsigned char> newDepthSource;

        newKeys.reserve(selected.size());
        newKeysUn.reserve(selected.size());
        newDepth.reserve(selected.size());
        newRight.reserve(selected.size());
        newIgevDepth.reserve(selected.size());
        newDepthConf.reserve(selected.size());
        newDepthSource.reserve(selected.size());

        cv::Mat newDescriptors;
        if (mDescriptors.rows == oldN && !mDescriptors.empty())
            newDescriptors.create((int)selected.size(), mDescriptors.cols, mDescriptors.type());

        std::vector<ORB_SLAM3::MapPoint*> newMapPoints;
        const bool hasMapPointVector = ((int)mvpMapPoints.size() == oldN);

        if (hasMapPointVector) newMapPoints.reserve(selected.size());

        for (int newIdx = 0; newIdx < (int)selected.size(); ++newIdx)
        {
            const int oldIdx = selected[newIdx];

            if (oldIdx < (int)mvKeys.size())
                newKeys.push_back(mvKeys[oldIdx]);

            if (oldIdx < (int)mvKeysUn.size())
                newKeysUn.push_back(mvKeysUn[oldIdx]);

            newDepth.push_back(mvDepth[oldIdx]);
            newRight.push_back(mvuRight[oldIdx]);
            newIgevDepth.push_back(mvIgevDepth[oldIdx]);
            newDepthConf.push_back(mvDepthConfidence[oldIdx]);
            newDepthSource.push_back(mvDepthSource[oldIdx]);

            if (mDescriptors.rows == oldN && !mDescriptors.empty())
                mDescriptors.row(oldIdx).copyTo(newDescriptors.row(newIdx));

            if (hasMapPointVector)
                newMapPoints.push_back(mvpMapPoints[oldIdx]);

        }

        if ((int)newKeys.size() == (int)selected.size())
            mvKeys.swap(newKeys);

        if ((int)newKeysUn.size() == (int)selected.size())
            mvKeysUn.swap(newKeysUn);

        mvDepth.swap(newDepth);
        mvuRight.swap(newRight);
        mvIgevDepth.swap(newIgevDepth);
        mvDepthConfidence.swap(newDepthConf);
        mvDepthSource.swap(newDepthSource);

        if (mDescriptors.rows == oldN && !mDescriptors.empty())
            mDescriptors = newDescriptors;

        if (hasMapPointVector)
            mvpMapPoints.swap(newMapPoints);


        N = (int)selected.size();
        selectedN = N;
    }

    int emptyCells = 0;
    int maxCellCount = 0;
    int minNonEmptyCellCount = std::numeric_limits<int>::max();

    for (int c = 0; c < nCells; ++c)
    {
        if (gridCount[c] == 0)
        {
            emptyCells++;
        }
        else
        {
            maxCellCount = std::max(maxCellCount, gridCount[c]);
            minNonEmptyCellCount = std::min(minNonEmptyCellCount, gridCount[c]);
        }
    }

    if (minNonEmptyCellCount == std::numeric_limits<int>::max())
        minNonEmptyCellCount = 0;

    if (evidenceFrameLogEnabled)
    {
        int evidenceSourceFinal[7] = {0, 0, 0, 0, 0, 0, 0};
        for (unsigned char source : mvDepthSource)
        {
            if (source < 7)
                evidenceSourceFinal[source]++;
        }

        float cellMean = nCells > 0 ? static_cast<float>(oldN) / nCells : 0.0f;
        float cellSqDiff = 0.0f;
        float gridEntropy = 0.0f;
        int occupiedCells = 0;

        for (int c = 0; c < nCells; ++c)
        {
            if (gridCount[c] > 0)
            {
                occupiedCells++;

                const float p =
                    oldN > 0 ? static_cast<float>(gridCount[c]) / oldN : 0.0f;
                if (p > 0.0f)
                    gridEntropy -= p * std::log(p);
            }

            const float diff = gridCount[c] - cellMean;
            cellSqDiff += diff * diff;
        }

        const float cellStd =
            nCells > 0 ? std::sqrt(cellSqDiff / nCells) : 0.0f;
        const float cellCv =
            cellMean > 1e-6f ? cellStd / cellMean : 0.0f;
        const float normalizedEntropy =
            nCells > 1 ? gridEntropy / std::log(static_cast<float>(nCells)) : 0.0f;

        std::ofstream evidence(evidenceCsvPath, std::ios::app);
        if (evidence.is_open())
        {
            const float avgConf = confCount > 0 ? confSum / confCount : 0.0f;
            evidence << std::fixed << mTimeStamp
                     << "," << oldN
                     << "," << selectedN
                     << "," << injected
                     << "," << replaced_high
                     << "," << replaced_mid
                     << "," << kept_original
                     << "," << conflict_keep_stereo
                     << "," << rejected
                     << "," << invalid_count
                     << "," << igev_only_suppressed
                     << "," << igev_only_mono
                     << "," << edge_penalty
                     << "," << evidenceSourceFinal[0]
                     << "," << evidenceSourceFinal[1]
                     << "," << evidenceSourceFinal[2]
                     << "," << evidenceSourceFinal[3]
                     << "," << evidenceSourceFinal[4]
                     << "," << evidenceSourceFinal[5]
                     << "," << evidenceSourceFinal[6]
                     << "," << igevCalibRawPairs
                     << "," << igevCalibUsedPairs
                     << "," << igevCalibRawScale
                     << "," << igevCalibAppliedScale
                     << "," << (igevCalibApplied ? 1 : 0)
                     << "," << igevCalibMedianRelBefore
                     << "," << igevCalibMedianRelAfter
                     << "," << igevCalibP90RelBefore
                     << "," << igevCalibP90RelAfter
                     << "," << avgConf
                     << "," << occupiedCells
                     << "," << emptyCells
                     << "," << minNonEmptyCellCount
                     << "," << maxCellCount
                     << "," << cellMean
                     << "," << cellStd
                     << "," << cellCv
                     << "," << normalizedEntropy
                     << "," << (conservativeQuotaDecisions > 0
                                     ? static_cast<float>(conservativeQuotaSum) /
                                           conservativeQuotaDecisions
                                     : 0.0f)
                     << "," << adaptiveQuotaSuppressed
                     << "," << recoveryCellCount
                     << std::endl;
        }
    }

    const char* pointSourceCsvPath = std::getenv("ROVER_POINT_SOURCE_CSV");
    if (pointSourceCsvPath && pointSourceCsvPath[0] != '\0')
    {
        bool shouldWritePointSources = true;
        const char* targetTsEnv = std::getenv("ROVER_POINT_SOURCE_TARGET_TS");
        if (targetTsEnv && targetTsEnv[0] != '\0')
        {
            const double targetTs = std::atof(targetTsEnv);
            const char* tolEnv = std::getenv("ROVER_POINT_SOURCE_TOL");
            const double tol = (tolEnv && tolEnv[0] != '\0') ? std::atof(tolEnv) : 1e-3;
            shouldWritePointSources = std::fabs(mTimeStamp - targetTs) <= tol;
        }

        if (shouldWritePointSources)
        {
            std::ifstream existing(pointSourceCsvPath);
            const bool needHeader = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
            existing.close();

            std::ofstream pointCsv(pointSourceCsvPath, std::ios::app);
            if (pointCsv.is_open())
            {
                if (needHeader)
                {
                    pointCsv << "frame_ts,feature_id,x,y,source_id,source_class,selected,"
                             << "stereo_depth,igev_depth,final_depth,u_right,depth_confidence"
                             << std::endl;
                }

                auto sourceName = [](unsigned char source) -> const char*
                {
                    switch (source)
                    {
                    case 0:
                        return "STEREO_ONLY";
                    case 1:
                        return "IGEV_ONLY";
                    case 2:
                        return "FUSED_HIGH";
                    case 3:
                        return "CONFLICT_STEREO";
                    case 4:
                        return "INVALID";
                    case 5:
                        return "FUSED_MID";
                    case 6:
                        return "SUPPRESSED";
                    default:
                        return "UNKNOWN";
                    }
                };

                const int nPointRows = std::min(selectedN, static_cast<int>(mvKeysUn.size()));
                for (int i = 0; i < nPointRows; ++i)
                {
                    const unsigned char source =
                        i < static_cast<int>(mvDepthSource.size()) ? mvDepthSource[i] : 4;
                    const float igevDepthValue =
                        i < static_cast<int>(mvIgevDepth.size()) ? mvIgevDepth[i] : -1.0f;
                    const float depthConfValue =
                        i < static_cast<int>(mvDepthConfidence.size()) ? mvDepthConfidence[i] : 0.0f;
                    const float uRightValue =
                        i < static_cast<int>(mvuRight.size()) ? mvuRight[i] : -1.0f;
                    const float finalDepthValue =
                        i < static_cast<int>(mvDepth.size()) ? mvDepth[i] : -1.0f;

                    float stereoDepthValue = -1.0f;
                    if (uRightValue >= 0.0f)
                    {
                        const float disparity = mvKeysUn[i].pt.x - uRightValue;
                        if (disparity > 1e-6f)
                            stereoDepthValue = mbf / disparity;
                    }

                    pointCsv << std::fixed << mTimeStamp
                             << "," << i
                             << "," << mvKeysUn[i].pt.x
                             << "," << mvKeysUn[i].pt.y
                             << "," << static_cast<int>(source)
                             << "," << sourceName(source)
                             << "," << 1
                             << "," << stereoDepthValue
                             << "," << igevDepthValue
                             << "," << finalDepthValue
                             << "," << uRightValue
                             << "," << depthConfValue
                             << std::endl;
                }
            }
        }
    }

    if (profileMonitorLogEnabled)
    {
        std::vector<int> acceptedIgevOnlyGrid(nCells, 0);
        int acceptedIgevOnlyOccupiedCells = 0;
        int acceptedIgevOnlyMaxCell = 0;
        float acceptedIgevOnlyEntropy = 0.0f;
        float acceptedIgevOnlyCellSqDiff = 0.0f;

        for (int i = 0; i < selectedN && i < (int)mvKeysUn.size(); ++i)
        {
            if (i >= (int)mvDepthSource.size() || mvDepthSource[i] != 1)
                continue;

            const cv::KeyPoint& kp = mvKeysUn[i];
            const int gx = std::min(
                gridCols - 1,
                std::max(0, (int)(kp.pt.x * gridCols / std::max(1, igevDepth.cols))));
            const int gy = std::min(
                gridRows - 1,
                std::max(0, (int)(kp.pt.y * gridRows / std::max(1, igevDepth.rows))));
            const int cell = gy * gridCols + gx;

            if (cell >= 0 && cell < nCells)
                acceptedIgevOnlyGrid[cell]++;
        }

        for (int c = 0; c < nCells; ++c)
        {
            if (acceptedIgevOnlyGrid[c] > 0)
            {
                acceptedIgevOnlyOccupiedCells++;
                acceptedIgevOnlyMaxCell =
                    std::max(acceptedIgevOnlyMaxCell, acceptedIgevOnlyGrid[c]);

                const float p =
                    injected > 0
                        ? static_cast<float>(acceptedIgevOnlyGrid[c]) / injected
                        : 0.0f;
                if (p > 0.0f)
                    acceptedIgevOnlyEntropy -= p * std::log(p);
            }

            const float mean =
                nCells > 0 ? static_cast<float>(injected) / nCells : 0.0f;
            const float diff = acceptedIgevOnlyGrid[c] - mean;
            acceptedIgevOnlyCellSqDiff += diff * diff;
        }

        const float acceptedIgevOnlyCellMean =
            nCells > 0 ? static_cast<float>(injected) / nCells : 0.0f;
        const float acceptedIgevOnlyCellStd =
            nCells > 0 ? std::sqrt(acceptedIgevOnlyCellSqDiff / nCells) : 0.0f;
        const float acceptedIgevOnlyCellCv =
            acceptedIgevOnlyCellMean > 1e-6f
                ? acceptedIgevOnlyCellStd / acceptedIgevOnlyCellMean
                : 0.0f;
        const float acceptedIgevOnlyEntropyNorm =
            nCells > 1
                ? acceptedIgevOnlyEntropy / std::log(static_cast<float>(nCells))
                : 0.0f;

        const float featureCount = oldN > 0 ? static_cast<float>(oldN) : 1.0f;
        const float stereoValidRatio = stereoValidDepthCount / featureCount;
        const float stereoLowDispRatio =
            stereoValidDepthCount > 0
                ? static_cast<float>(stereoLowDisparityCount) / stereoValidDepthCount
                : 0.0f;
        const float stereoMedianDispProxy =
            stereoValidDepthCount > 0 ? stereoDisparitySum / stereoValidDepthCount : 0.0f;
        const float igevValidRatio = igevValidDepthCount / featureCount;
        const float bothDepthRatio = bothDepthCount / featureCount;
        const float igevOnlyCandidateRatio = igevOnlyCandidateCount / featureCount;
        const float igevOnlyEligibleRatio = igevOnlyEligibleCount / featureCount;
        const float igevOnlyAcceptedRatio = injected / featureCount;
        const float highRatio = replaced_high / featureCount;
        const float midRatio = replaced_mid / featureCount;
        const float conflictRatio = conflict_count / featureCount;
        const float edgeRatio = edge_penalty / featureCount;
        const float effectiveWeightMean =
            !effectiveIgevWeights.empty()
                ? effectiveIgevWeightSum / effectiveIgevWeights.size()
                : 0.0f;
        const float effectiveWeightP50 = percentileValue(effectiveIgevWeights, 0.50f);
        const float effectiveWeightP90 = percentileValue(effectiveIgevWeights, 0.90f);
        const float quotaAvg =
            conservativeQuotaDecisions > 0
                ? static_cast<float>(conservativeQuotaSum) / conservativeQuotaDecisions
                : 0.0f;
        const float quotaUtilization =
            adaptiveQuotaAllowedSum > 0
                ? static_cast<float>(injected) / adaptiveQuotaAllowedSum
                : 0.0f;

        std::ofstream monitor(profileMonitorCsvPath, std::ios::app);
        if (monitor.is_open())
        {
            monitor << std::fixed
                    << (evidenceProfileName ? evidenceProfileName : "")
                    << "," << (evidenceVariantName ? evidenceVariantName : "")
                    << "," << mTimeStamp
                    << "," << oldN
                    << "," << selectedN
                    << "," << stereoValidDepthCount
                    << "," << stereoValidRatio
                    << "," << stereoLowDisparityCount
                    << "," << stereoLowDispRatio
                    << "," << stereoMedianDispProxy
                    << "," << igevValidDepthCount
                    << "," << igevValidRatio
                    << "," << bothDepthCount
                    << "," << bothDepthRatio
                    << "," << igevOnlyCandidateCount
                    << "," << igevOnlyCandidateRatio
                    << "," << igevOnlyEligibleCount
                    << "," << igevOnlyEligibleRatio
                    << "," << injected
                    << "," << igevOnlyAcceptedRatio
                    << "," << igev_only_suppressed
                    << "," << replaced_high
                    << "," << highRatio
                    << "," << replaced_mid
                    << "," << midRatio
                    << "," << conflict_count
                    << "," << conflictRatio
                    << "," << edge_penalty
                    << "," << edgeRatio
                    << "," << effectiveWeightMean
                    << "," << effectiveWeightP50
                    << "," << effectiveWeightP90
                    << "," << igevCalibRawPairs
                    << "," << igevCalibUsedPairs
                    << "," << igevCalibAppliedScale
                    << "," << (igevCalibApplied ? 1 : 0)
                    << "," << igevCalibMedianRelBefore
                    << "," << igevCalibMedianRelAfter
                    << "," << igevCalibP90RelBefore
                    << "," << igevCalibP90RelAfter
                    << "," << quotaAvg
                    << "," << quotaUtilization
                    << "," << acceptedIgevOnlyOccupiedCells
                    << "," << (nCells - acceptedIgevOnlyOccupiedCells)
                    << "," << acceptedIgevOnlyEntropyNorm
                    << "," << acceptedIgevOnlyCellCv
                    << "," << acceptedIgevOnlyMaxCell
                    << "," << useContinuousFrameWeight
                    << "," << continuousStereoRatioEma
                    << "," << continuousSceneDemand
                    << "," << continuousHighWeightCommanded
                    << "," << continuousMidWeightCommanded
                    << "," << continuousHighEdgeWeightCommanded
                    << "," << continuousMidEdgeWeightCommanded
                    << "," << continuousDemandSaturatedLight
                    << "," << continuousDemandSaturatedRobust
                    << std::endl;
        }
    }

    static int print_count = 0;
    if (print_count < 100 || print_count % 100 == 0)
    {
        const float avgConf = confCount > 0 ? confSum / confCount : 0.0f;

        std::cout << "[IGEV D4 REAL] frame_ts=" << std::fixed << mTimeStamp
                  << " oldN=" << oldN
                  << " selectedN=" << selectedN
                  << " grid_select=" << (gridSelectionEnabled ? 1 : 0)
                  << " grid=" << gridCols << "x" << gridRows
                  << " injected=" << injected
                  << " replaced_high=" << replaced_high
                  << " replaced_mid=" << replaced_mid
                  << " kept_original=" << kept_original
                  << " conflict_keep_stereo=" << conflict_keep_stereo
                  << " rejected=" << rejected
                  << " edge_penalty=" << edge_penalty
                  << " consistent=" << consistent_count
                  << " conflict=" << conflict_count
                  << " igev_only=" << igev_only_count
                  << " stereo_only=" << stereo_only_count
                  << " invalid=" << invalid_count
                  << " igev_only_suppressed=" << igev_only_suppressed
                  << " igev_only_mono=" << igev_only_mono
                  << " mono_residual=" << igevOnlyMonoResidual
                  << " fuse50_enabled=" << useIgevFuse50
                  << " fuse50_weight=" << igevFuseWeight
                  << " fuse50_high=" << fuse50_high
                  << " fuse50_mid=" << fuse50_mid
                  << " fuse_high_weight=" << igevFuseHighWeight
                  << " fuse_mid_weight=" << igevFuseMidWeight
                  << " fuse_high_count=" << fuse50_high
                  << " fuse_mid_count=" << fuse50_mid
                  << " continuous_frame_weight=" << useContinuousFrameWeight
                  << " continuous_raw_stereo_ratio=" << continuousStereoRatioRaw
                  << " continuous_ema_stereo_ratio=" << continuousStereoRatioEma
                  << " continuous_scene_demand=" << continuousSceneDemand
                  << " continuous_high_weight=" << continuousHighWeightCommanded
                  << " continuous_mid_weight=" << continuousMidWeightCommanded
                  << " continuous_confidence_fusion=" << useContinuousConfidenceFusion
                  << " continuous_mid_confidence_only=" << useContinuousMidConfidenceOnly
                  << " continuous_confidence_tau=" << continuousConfidenceTau
                  << " continuous_confidence_avg="
                  << (continuousConfidenceCount > 0
                          ? continuousConfidenceSum / continuousConfidenceCount
                          : 0.0f)
                  << " continuous_confidence_high_avg="
                  << (continuousConfidenceHighCount > 0
                          ? continuousConfidenceHighSum / continuousConfidenceHighCount
                          : 0.0f)
                  << " continuous_confidence_mid_avg="
                  << (continuousConfidenceMidCount > 0
                          ? continuousConfidenceMidSum / continuousConfidenceMidCount
                          : 0.0f)
                  << " continuous_final_weight_avg="
                  << (continuousConfidenceCount > 0
                          ? continuousFinalWeightSum / continuousConfidenceCount
                          : 0.0f)
                  << " adaptive_fuse_enabled=" << useIgevAdaptiveFuse
                  << " adaptive_min_weight=" << igevAdaptiveMinWeight
                  << " adaptive_max_weight=" << igevAdaptiveMaxWeight
                  << " adaptive_avg_weight="
                  << (adaptiveFuseCount > 0
                          ? adaptiveWeightSum / adaptiveFuseCount
                          : 0.0f)
                  << " adaptive_fuse_count=" << adaptiveFuseCount
                  << " adaptive_edge_count=" << adaptiveEdgeCount
                  << " geometry_conf_fusion=" << useGeometryConfidenceFusion
                  << " geometry_avg_weight="
                  << (geometryFusionCount > 0
                          ? geometryFusionWeightSum / geometryFusionCount
                          : 0.0f)
                  << " geometry_count=" << geometryFusionCount
                  << " geometry_high=" << geometryFusionHighCount
                  << " geometry_mid=" << geometryFusionMidCount
                  << " geometry_edge=" << geometryFusionEdgeCount
                  << " geometry_strong_stereo=" << geometryFusionStrongStereoCount
                  << " geometry_weak_stereo=" << geometryFusionWeakStereoCount
                  << " geometry_min_weight=" << geometryFusionMinWeight
                  << " geometry_max_weight=" << geometryFusionMaxWeight
                  << " geometry_strong_disp=" << geometryStrongStereoDisp
                  << " geometry_mid_disp=" << geometryMidStereoDisp
                  << " geometry_weak_disp=" << geometryWeakStereoDisp
                  << " edge_aware_enabled=" << useIgevEdgeAwareFuse
                  << " high_edge_weight=" << igevFuseHighEdgeWeight
                  << " mid_edge_weight=" << igevFuseMidEdgeWeight
                  << " edge_aware_high=" << edgeAwareHighCount
                  << " edge_aware_mid=" << edgeAwareMidCount
                  << " adaptive_igev_only_quota=" << useAdaptiveIgevOnlyQuota
                  << " target_depth_per_cell=" << targetDepthPerCell
                  << " igev_only_per_cell_max=" << adaptiveIgevOnlyPerCellMax
                  << " conservative_quota=" << useConservativeIgevOnlyQuota
                  << " quota_min=" << conservativeQuotaMin
                  << " quota_max=" << conservativeQuotaMax
                  << " conservative_quota_avg="
                  << (conservativeQuotaDecisions > 0
                          ? static_cast<float>(conservativeQuotaSum) /
                                conservativeQuotaDecisions
                          : 0.0f)
                  << " conservative_quota_min_count=" << conservativeQuotaMinCount
                  << " conservative_quota_mid_count=" << conservativeQuotaMidCount
                  << " conservative_quota_max_count=" << conservativeQuotaMaxCount
                  << " risk_gate=" << useIgevOnlyRiskGate
                  << " risk_min_disp=" << igevOnlyMinDisparity
                  << " risk_far_disp=" << igevOnlyFarDisparity
                  << " risk_far_min_score=" << igevOnlyFarMinScore
                  << " risk_local_var_gate=" << useIgevOnlyLocalVarGate
                  << " risk_local_rel_max=" << igevOnlyLocalRelMax
                  << " risk_pass=" << riskGatePass
                  << " risk_reject_disp=" << riskGateRejectDisparity
                  << " risk_reject_score=" << riskGateRejectScore
                  << " risk_reject_var=" << riskGateRejectLocalVar
                  << " risk_avg_local_rel="
                  << (riskGateLocalRelCount > 0
                          ? riskGateLocalRelSum / riskGateLocalRelCount
                          : 0.0f)
                  << " need_cell_recovery=" << useNeedBasedCellRecovery
                  << " overflow_recovery=" << useOverflowAwareRecovery
                  << " recovery_cells=" << recoveryCellCount
                  << " recovery_plus1=" << recoveryPlusOneCount
                  << " recovery_full8=" << recoveryFullEightCount
                  << " recovery_veto_edge=" << recoveryVetoEdge
                  << " recovery_veto_conflict=" << recoveryVetoConflict
                  << " recovery_sparse_cells=" << recoveryStereoSparseCells
                  << " recovery_empty_cells=" << recoveryDepthEmptyCells
                  << " recovery_eligible_cells=" << recoveryEligibleCells
                  << " recovery_overflow_cells=" << recoveryOverflowCells
                  << " recovery_overflow_candidates=" << recoveryOverflowCandidates
                  << " recovery_no_room_cells=" << recoveryNoRoomCells
                  << " recovery_high_score_overflow_cells=" << recoveryHighScoreOverflowCells
                  << " recovery_base_quota_avg="
                  << (nCells > 0
                          ? static_cast<float>(recoveryBaseQuotaSum) / nCells
                          : 0.0f)
                  << " recovery_final_quota_avg="
                  << (nCells > 0
                          ? static_cast<float>(recoveryFinalQuotaSum) / nCells
                          : 0.0f)
                  << " adaptive_quota_allowed_sum=" << adaptiveQuotaAllowedSum
                  << " adaptive_quota_suppressed=" << adaptiveQuotaSuppressed
                  << " adaptive_calib_conf="
                  << (igevCalibApplied
                          ? std::max(
                                0.0f,
                                std::min(
                                    1.0f,
                                    static_cast<float>(igevCalibUsedPairs) / 100.0f))
                          : 0.0f)
                  << " avg_conf=" << avgConf
                  << " disparity_stats=" << useDisparityStats
                  << " disp_ge10=" << disparityBinGe10
                  << " disp_5_10=" << disparityBin5To10
                  << " disp_2_5=" << disparityBin2To5
                  << " disp_1_2=" << disparityBin1To2
                  << " disp_lt1=" << disparityBinLt1
                  << " mean_rel_diff="
                  << (relDiffCount > 0 ? relDiffSum / relDiffCount : 0.0f)
                  << " conflict_disp_stats=" << useConflictDisparityStats
                  << " conflict_disp_ge10=" << conflictDispGe10
                  << " conflict_disp_5_10=" << conflictDisp5To10
                  << " conflict_disp_2_5=" << conflictDisp2To5
                  << " conflict_disp_lt2=" << conflictDispLt2
                  << " conflict_edge_ge10=" << conflictEdgeGe10
                  << " conflict_edge_5_10=" << conflictEdge5To10
                  << " conflict_edge_2_5=" << conflictEdge2To5
                  << " conflict_edge_lt2=" << conflictEdgeLt2
                  << " conflict_rel_ge10="
                  << (conflictDispGe10 > 0
                          ? conflictRelGe10 / conflictDispGe10
                          : 0.0f)
                  << " conflict_rel_5_10="
                  << (conflictDisp5To10 > 0
                          ? conflictRel5To10 / conflictDisp5To10
                          : 0.0f)
                  << " conflict_rel_2_5="
                  << (conflictDisp2To5 > 0
                          ? conflictRel2To5 / conflictDisp2To5
                          : 0.0f)
                  << " conflict_rel_lt2="
                  << (conflictDispLt2 > 0
                          ? conflictRelLt2 / conflictDispLt2
                          : 0.0f)
                  << " conflict_score_ge10="
                  << (conflictDispGe10 > 0
                          ? conflictScoreGe10 / conflictDispGe10
                          : 0.0f)
                  << " conflict_score_5_10="
                  << (conflictDisp5To10 > 0
                          ? conflictScore5To10 / conflictDisp5To10
                          : 0.0f)
                  << " conflict_score_2_5="
                  << (conflictDisp2To5 > 0
                          ? conflictScore2To5 / conflictDisp2To5
                          : 0.0f)
                  << " conflict_score_lt2="
                  << (conflictDispLt2 > 0
                          ? conflictScoreLt2 / conflictDispLt2
                          : 0.0f)
                  << " empty_cells=" << emptyCells
                  << " min_cell=" << minNonEmptyCellCount
                  << " max_cell=" << maxCellCount
                  << " depth_shape=" << igevDepth.cols << "x" << igevDepth.rows
                  << std::endl;
    }

    print_count++;
}
