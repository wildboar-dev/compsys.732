//--------------------------------------------------
// Startup code module
//
// @author: Wild Boar
//
// @date: 2025-03-30
//--------------------------------------------------

#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
using namespace std;

#include <NVLib/Logger.h>
#include <NVLib/Math3D.h>
#include <NVLib/PoseUtils.h>
#include <NVLib/SaveUtils.h>
#include <NVLib/PlaneUtils.h>
#include <NVLib/CloudUtils.h>
#include <NVLib/LoadUtils.h>
#include <NVLib/Path/PathHelper.h>
#include <NVLib/Parameters/Parameters.h>

#include <NVLib/Odometry/FastDetector.h>
#include <NVLib/Odometry/FastTracker.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include "ArgReader.h"

//--------------------------------------------------
// Function Prototypes
//--------------------------------------------------
void Run(NVLib::Parameters * parameters);
Mat LoadDepth(NVLib::MonoCalibration * calibration, const string& path);
Mat RenderFeatures(const Mat& image, const vector<KeyPoint>& keypoints);
Mat RenderMatches(NVLib::StereoFrame& frame, vector<NVLib::MatchIndices *>& matches, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2);
float ExtractDepth(Mat& depth, const Point2f& location);
Mat EstimatePose(Mat& camera, vector<Point3f>& scenePoints, vector<Point2f>& imagePoints);
void EstimateError(Mat& camera, Mat& pose, vector<Point3f>& scenePoints, vector<Point2f>& imagePoints, Vec2d& error); 

//--------------------------------------------------
// Execution Logic
//--------------------------------------------------

/**
 * Main entry point into the application
 * @param parameters The input parameters
 */
void Run(NVLib::Parameters * parameters) 
{
    if (parameters == nullptr) return; auto logger = NVLib::Logger(1);

    logger.StartApplication();

    logger.Log(1, "Loading calibration");
    auto calibration = NVLib::LoadUtils::LoadCalibration("calibration.xml");

    logger.Log(1, "Loading the first frame");
    Mat color_1 = imread("color_0000.png"); if (color_1.empty()) throw runtime_error("Unable to open color_0000.png");
    Mat uColor_1; undistort(color_1, uColor_1, calibration->GetCamera(), calibration->GetDistortion());
    Mat depth_1 = LoadDepth(calibration, "depth_0000.png");
    Mat fdepth_1; depth_1.convertTo(fdepth_1, CV_32F);
    Mat cloud_1 = NVLib::CloudUtils::BuildColorCloud(calibration->GetCamera(), uColor_1, depth_1);

    logger.Log(1, "Loading the second frame");
    Mat color_2 = imread("color_0001.png"); if (color_2.empty()) throw runtime_error("Unable to open color_0001.png");
    Mat uColor_2; undistort(color_2, uColor_2, calibration->GetCamera(), calibration->GetDistortion());
    Mat depth_2 = LoadDepth(calibration, "depth_0001.png");
    Mat fdepth_2; depth_2.convertTo(fdepth_2, CV_32F);
    Mat cloud_2 = NVLib::CloudUtils::BuildColorCloud(calibration->GetCamera(), uColor_2, depth_2);

    logger.Log(1, "Detecting features in the first frame");
    auto detector = NVLib::FastDetector(3);
    auto keypoints_1 = vector<KeyPoint>(); detector.Extract(uColor_1, keypoints_1);

    logger.Log(1, "Detecting features in the second frame");
    detector.SetFrame(uColor_1, uColor_2);
    auto matches = vector<NVLib::MatchIndices*>();
    auto keypoints_2 = vector<KeyPoint>(); detector.Extract(uColor_2, keypoints_2);
    detector.Match(keypoints_1, keypoints_2, matches);

    auto stereoFrame = NVLib::StereoFrame(uColor_1, uColor_2);
    Mat display = RenderMatches(stereoFrame, matches, keypoints_1, keypoints_2);
    imwrite("Output/matches.png", display);

    logger.Log(1, "Extracting scene points");
    auto scenePoints = vector<Point3f>();  auto imagePoints = vector<Point2f>(); 
    for (auto match : matches) 
	{
		// Retrieve image points from the system
		auto point = keypoints_1[match->GetFirstId()].pt;	

		// Get the depth from the system
		auto Z = ExtractDepth(fdepth_1, point);

		// Handle the error case
		if (Z <= 0) continue;

		// Defines the parameters that make up the variables
		auto cdata = (double *) calibration->GetCamera().data;
		auto fx = cdata[0]; auto fy = cdata[4];
		auto cx = cdata[2]; auto cy = cdata[5];

		// Convert to a 3D point
		auto X = (point.x - cx) * (Z / fx);
		auto Y = (point.y - cy) * (Z / fy);
		
		// Add the 3D point to the collection
		scenePoints.push_back(Point3f(X, Y, Z)); 

        // Add the image point to the collection
        auto point2 = keypoints_2[match->GetSecondId()].pt;
        imagePoints.push_back(point2);
    }        

    logger.Log(1, "Estimating the pose");
    Mat pose = EstimatePose(calibration->GetCamera(), scenePoints, imagePoints);

    logger.Log(1, "Determining the reprojection error"); 
    auto error = Vec2d(); EstimateError(calibration->GetCamera(), pose, scenePoints, imagePoints, error);
    logger.Log(1, "Mean Error: %f +/- %f", error[0], error[1]);

    logger.Log(1, "Saving the point clouds");
    Mat invPose = pose.inv();
    Mat tcloud_2 = NVLib::CloudUtils::TransformCloud(cloud_2, invPose);
    NVLib::CloudUtils::Save("Output/cloud_1.ply", cloud_1);
    NVLib::CloudUtils::Save("Output/cloud_2.ply", cloud_2);
    NVLib::CloudUtils::Save("Output/tcloud_2.ply", tcloud_2);


    logger.Log(1, "Free Data");
    for (auto match : matches) delete match; 
    delete calibration;


    logger.StopApplication();
}

//--------------------------------------------------
// Loading Helpers
//--------------------------------------------------

/**
 * Loader functionality
 * @param calibration The calibration object
 * @param path The path to the depth image
 */
Mat LoadDepth(NVLib::MonoCalibration * calibration, const string& path) 
{
    Mat depth = imread(path, IMREAD_UNCHANGED);
    if (depth.empty()) throw runtime_error("Unable to open " + path);
    
    Mat camera = calibration->GetCamera();
    Mat distortion = calibration->GetDistortion();
    Mat R = Mat::eye(3, 3, CV_64F);

    Mat mapx, mapy; initUndistortRectifyMap(camera, distortion, R, camera, depth.size(), CV_32FC1, mapx, mapy);
    Mat uDepth; remap(depth, uDepth, mapx, mapy, INTER_NEAREST);

    Mat result; uDepth.convertTo(result, CV_64F);

    return result;
}

//--------------------------------------------------
// Render Logic
//--------------------------------------------------

/**
 * Render the features on the image
 * @param image The image to render on
 * @param keypoints The keypoints to render
 * @return The image with the keypoints rendered
 */
Mat RenderFeatures(const Mat& image, const vector<KeyPoint>& keypoints) 
{
    Mat result = image.clone();
    for (const auto& keypoint : keypoints) 
    {
        circle(result, keypoint.pt, 3, Scalar(197, 255, 255), -1);
        circle(result, keypoint.pt, 3, Scalar::all(0), 2);
    }
    return result;
}

/**
 * Render the matches on the image
 * @param frame The stereo frame
 * @param matches The matches to render
 * @param keypoints_1 The keypoints from the first image
 * @param keypoints_2 The keypoints from the second image
 * @return The image with the matches rendered
 */
Mat RenderMatches(NVLib::StereoFrame& frame, vector<NVLib::MatchIndices *>& matches, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2) 
{
    auto displayMatches = vector<NVLib::FeatureMatch>();
	for (auto& match : matches) 
	{	
		auto id_1 = match->GetFirstId(); auto id_2 = match->GetSecondId();
		auto m = NVLib::FeatureMatch(keypoints_1[id_1].pt, keypoints_2[id_2].pt);
		displayMatches.push_back(m);
	}

    int actualWidth = frame.GetLeft().cols + frame.GetRight().cols;
	auto factor = (double)1000 / (double)actualWidth;

	Mat result = Mat::zeros(frame.GetLeft().rows, actualWidth, CV_8UC3);
	int height1 = frame.GetLeft().rows; int width1 = frame.GetLeft().cols;
	frame.GetLeft().copyTo(result(Range(0, height1), Range(0, width1)));

	int height2 = frame.GetRight().rows; int width2 = frame.GetRight().cols;
	frame.GetRight().copyTo(result(Range(0, height2), Range(width1, width1 + width2)));

	Mat smallImage; resize(result, smallImage, Size(), factor, factor);

	auto displayColor = Scalar(0, 0, 255);

	for (auto& match : displayMatches) 
	{
		auto point = match.GetPoint1();
		auto x = (int)round(point.x * factor);
		auto y = (int)round(point.y * factor);
		circle(smallImage, Point(x, y), 3, displayColor, FILLED);
		circle(smallImage, Point(x, y), 4, Scalar(), 1);
	}

	for (auto& match : displayMatches)
	{
		auto point = match.GetPoint2();
		auto x = (int)round(((double)point.x + width1) * factor);
		auto y = (int)round(point.y * factor);
		circle(smallImage, Point(x, y), 3, displayColor, -1);
		circle(smallImage, Point(x, y), 4, Scalar(), 1);
	}

    return smallImage;
}

//--------------------------------------------------
// Extra Points
//--------------------------------------------------

/**
 * @brief Add the logic to extract depth from a given system
 * @param depth The depth value that we are extracting
 * @param location The location of the depth value that we are extracting
 * @return float The depth value that we have gotten from the file
 */
float ExtractDepth(Mat& depth, const Point2f& location) 
{
	auto x = (int)round(location.x); auto y = (int)round(location.y);
	if (x < 0 || y < 0 || x >= depth.cols || y >= depth.rows) return 0;
	auto data = (float *) depth.data; auto index = x + y * depth.cols;
	return data[index];
}

//--------------------------------------------------
// Pose Estimation
//--------------------------------------------------

/**
 * @brief Perform the pose estimation logic
 * @param camera The given camera matrix
 * @param scenePoints The list of scene points
 * @param imagePoints The list of image points
 * @return Mat The pose that was estimated
 */
Mat EstimatePose(Mat& camera, vector<Point3f>& scenePoints, vector<Point2f>& imagePoints) 
{
	// Convert the scene points and image points to doubles
	auto dscene = vector<Point3d>(); auto dimage = vector<Point2d>();
	for (auto i = 0; i < scenePoints.size(); i++) 
	{
		dscene.push_back(Point3d(scenePoints[i].x, scenePoints[i].y, scenePoints[i].z)); 
		dimage.push_back(Point2d(imagePoints[i].x, imagePoints[i].y));
	}

	// Perform the pose estimation
	Mat nodistortion = Mat_<double>::zeros(4,1);
	Vec3d rvec, tvec; solvePnPRansac(dscene, dimage, camera, nodistortion, rvec, tvec, false, 1e6, 5, 0.9, noArray(), SOLVEPNP_DLS);

	// Return the result
	return NVLib::PoseUtils::Vectors2Pose(rvec, tvec);
}

//--------------------------------------------------
// Error Estimation
//--------------------------------------------------

/**
 * @brief Determine the reprojection error associated with the pose
 * @param camera The given camera matrix
 * @param pose The estimated pose
 * @param scenePoints The list of scene points
 * @param imagePoints The list of image points
 * @param error The error point that we are getting
 */
void EstimateError(Mat& camera, Mat& pose, vector<Point3f>& scenePoints, vector<Point2f>& imagePoints, Vec2d& error) 
{
	// Convert the scene points and image points to doubles
	auto dscene = vector<Point3d>(); auto dimage = vector<Point2d>();
	for (auto i = 0; i < scenePoints.size(); i++) 
	{
		dscene.push_back(Point3d(scenePoints[i].x, scenePoints[i].y, scenePoints[i].z)); 
		dimage.push_back(Point2d(imagePoints[i].x, imagePoints[i].y));
	}

	// Convert the pose matrix to some vectors
	auto rvec = Vec3d(); auto tvec = Vec3d(); NVLib::PoseUtils::Pose2Vectors(pose, rvec, tvec);

	// Project 3D points to get "estimated points"
	Mat nodistortion = Mat_<double>::zeros(4,1);
	auto estimated = vector<Point2d>(); projectPoints(dscene, rvec, tvec, camera, nodistortion, estimated);

	// Calculate the errors
	auto errors = vector<double>();
    for (auto i = 0; i < estimated.size(); i++) 
    {
        auto xDiff = dimage[i].x - estimated[i].x;
        auto yDiff = dimage[i].y - estimated[i].y;
        auto length = sqrt(xDiff * xDiff + yDiff * yDiff);
		errors.push_back(length);
    }

	// Extract the error "summaries"
	auto mean = Scalar(); auto stddev = Scalar();
	cv::meanStdDev(errors, mean, stddev);
	error[0] = mean[0]; error[1] = stddev[0];
}


//--------------------------------------------------
// Entry Point
//--------------------------------------------------

/**
 * Main Method
 * @param argc The count of the incoming arguments
 * @param argv The number of incoming arguments
 * @return SUCCESS and FAILURE
 */
int main(int argc, char ** argv) 
{
    NVLib::Parameters * parameters = nullptr;

    try
    {
        parameters = NVL_Utils::ArgReader::GetParameters(argc, argv);
        Run(parameters);
    }
    catch (runtime_error exception)
    {
        cerr << "Error: " << exception.what() << endl;
        exit(EXIT_FAILURE);
    }
    catch (string exception)
    {
        cerr << "Error: " << exception << endl;
        exit(EXIT_FAILURE);
    }

    if (parameters != nullptr) delete parameters;

    return EXIT_SUCCESS;
}