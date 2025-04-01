//--------------------------------------------------
// Implementation of class ColorMatcher
//
// @author: Wild Boar
//
// @date: 2024-08-15
//--------------------------------------------------

#include "ColorMatcher.h"
using namespace NVL_App;

//--------------------------------------------------
// Constructors and Terminators
//--------------------------------------------------

/**
 * @brief Default Constructor
 */
ColorMatcher::ColorMatcher()
{
    _iterCounts.push_back(7);
    _iterCounts.push_back(7);
    _iterCounts.push_back(7);
    _iterCounts.push_back(10);

    _minGradMagnitudes.push_back(12);
    _minGradMagnitudes.push_back(5);
    _minGradMagnitudes.push_back(3);
    _minGradMagnitudes.push_back(1);
}

//--------------------------------------------------
// Find Pose
//--------------------------------------------------

/**
 * @brief Find the given pose
 * @param camera The given camera matrix
 * @param initPose The initial pose tha we are working with
 * @param frame_1 The first frame
 * @param frame_2 The second frame
 * @return Mat Returns a Mat
 */
Mat ColorMatcher::FindPose(Mat camera, Mat initPose, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2)
{
	Mat grayImage0, grayImage1, depthFlt0, depthFlt1, mask0, mask1;
    cvtColor(frame_1->GetColor(), grayImage0, COLOR_BGR2GRAY);
    cvtColor(frame_2->GetColor(), grayImage1, COLOR_BGR2GRAY);
    frame_1->GetDepth().convertTo(depthFlt0, CV_32FC1);
    frame_2->GetDepth().convertTo(depthFlt1, CV_32FC1);
	mask0 = frame_1->GetMask(); mask1 = frame_2->GetMask();
	
	Mat result; 

	auto odometryFinder = rgbd::RgbdICPOdometry(camera, minDepth, maxDepth, rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(), rgbd::Odometry::DEFAULT_MAX_POINTS_PART(), _iterCounts, _minGradMagnitudes);
	auto success = odometryFinder.compute(grayImage1, depthFlt1, mask1, grayImage0, depthFlt0, mask0, result, initPose);

    cout << "Success: " << (success ? "true" : "false") << endl;

	return result;
}