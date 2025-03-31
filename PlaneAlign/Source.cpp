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

#include <opencv2/opencv.hpp>
using namespace cv;

#include "ArgReader.h"

//--------------------------------------------------
// Function Prototypes
//--------------------------------------------------
void Run(NVLib::Parameters * parameters);
Mat LoadDepth(NVLib::MonoCalibration * calibration, const string& path);
Mat GetGradient(NVLib::MonoCalibration * calibration, Mat& cloud);
Point3d ReadPoint(Mat& cloud, int row, int column);
Mat GetGradientImage(Mat& gradient);
Mat ExtractPart(Mat& cloud, Mat& gradient, double zMin, double zMax, int gradientMax, double gThresh = 0.8, bool center = false);  
Vec4d FitPlaneRansac(Mat& cloud, Mat& mask, int iterations = 1000);
void SavePlane(const string& path, const Vec4d& plane, const Vec3d& color);  
Point2d FindMatch(Mat& image_1, Mat& image_2, const Point2d& point_1);
Point3d RayTrace(NVLib::MonoCalibration * calibration, const Vec4d& plane, const Point2d& point);
Mat FindRotation(Vec4d& boxPlane, Vec4d& floorPlane);

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
    Mat cloud_1 = NVLib::CloudUtils::BuildColorCloud(calibration->GetCamera(), uColor_1, depth_1);
    Mat gradient_1 = GetGradient(calibration, cloud_1);

    logger.Log(1, "Loading the second frame");
    Mat color_2 = imread("color_0001.png"); if (color_2.empty()) throw runtime_error("Unable to open color_0001.png");
    Mat uColor_2; undistort(color_2, uColor_2, calibration->GetCamera(), calibration->GetDistortion());
    Mat depth_2 = LoadDepth(calibration, "depth_0001.png");
    Mat cloud_2 = NVLib::CloudUtils::BuildColorCloud(calibration->GetCamera(), uColor_2, depth_2);
    Mat gradient_2 = GetGradient(calibration, cloud_2);

    logger.Log(1, "Extracting Box Fronts");
    Mat boxFront_1 = ExtractPart(cloud_1, gradient_1, 970, 1080, 2, 0.9, true); //imwrite("Output/boxFront_1.png", boxFront_1);
    Mat boxFront_2 = ExtractPart(cloud_2, gradient_2, 970, 1080, 2, 0.9, true); //imwrite("Output/boxFront_2.png", boxFront_2);

    //logger.Log(1, "Extracting Floor Samples");
    Mat floorSample_1 = ExtractPart(cloud_1, gradient_1, 730, 900, 1, 0.1); //imwrite("Output/floorSample_1.png", floorSample_1);
    Mat floorSample_2 = ExtractPart(cloud_2, gradient_2, 730, 900, 1, 0.1); //imwrite("Output/floorSample_2.png", floorSample_2);

    logger.Log(1, "Extracting Box Planes");
    auto boxPlane_1 = FitPlaneRansac(cloud_1, boxFront_1, 1000);
    auto boxPlane_2 = FitPlaneRansac(cloud_2, boxFront_2, 1000);
    //cout << "Box Plane 1: " << boxPlane_1 << endl;
    //cout << "Box Plane 2: " << boxPlane_2 << endl;

    //logger.Log(1, "Extracting Floor Planes");
    auto floorPlane_1 = FitPlaneRansac(cloud_1, floorSample_1, 1000);
    auto floorPlane_2 = FitPlaneRansac(cloud_2, floorSample_2, 1000);
    //cout << "Floor Plane 1: " << floorPlane_1 << endl;
    //cout << "Floor Plane 2: " << floorPlane_2 << endl;

    logger.Log(1, "Finding corresponding points");
    auto matchPoint = FindMatch(uColor_1, uColor_2, Point2d(323, 166));

    logger.Log(1, "Ray tracing points");
    auto point_1 = RayTrace(calibration, boxPlane_1, Point2d(323, 166));
    auto point_2 = RayTrace(calibration, boxPlane_2, matchPoint);
    //auto z_1 = NVLib::Math3D::ExtractDepth(depth_1, Point2d(323, 166));
    //auto z_2 = NVLib::Math3D::ExtractDepth(depth_2, matchPoint);
    //auto diff_1 = point_1 - NVLib::Math3D::UnProject(calibration->GetCamera(), Point2d(323, 166), z_1);
    //auto diff_2 = point_2 - NVLib::Math3D::UnProject(calibration->GetCamera(), matchPoint, z_2);
    
    logger.Log(1, "Finding box normal rotation");
    auto boxNormal_1 = NVLib::Math3D::NormalizeVector(Vec3d(boxPlane_1[0], boxPlane_1[1], boxPlane_1[2]));
    auto boxNormal_2 = NVLib::Math3D::NormalizeVector(Vec3d(boxPlane_2[0], boxPlane_2[1], boxPlane_2[2]));
    auto axis = NVLib::Math3D::NormalizeVector(boxNormal_2.cross(boxNormal_1));
    auto angle = acos(boxNormal_1.dot(boxNormal_2));
    auto rvec = angle * axis;
    Mat R_b; Rodrigues(rvec, R_b); Mat rotation_b = NVLib::PoseUtils::GetPose(R_b, Vec3d(0, 0, 0));

    logger.Log(1, "Finding floor normal rotation");
    auto floorNormal_1 = NVLib::Math3D::NormalizeVector(Vec3d(floorPlane_1[0], floorPlane_1[1], floorPlane_1[2]));
    auto floorNormal_2 = NVLib::Math3D::NormalizeVector(Vec3d(floorPlane_2[0], floorPlane_2[1], floorPlane_2[2]));
    auto f_1 = NVLib::Math3D::NormalizeVector(floorNormal_1.cross(boxNormal_1));
    auto f_2 = NVLib::Math3D::NormalizeVector(floorNormal_2.cross(boxNormal_1));
    auto f_angle = acos(f_1.dot(f_2));
    auto f_rvec = boxNormal_1 * f_angle;
    Mat R_f; Rodrigues(f_rvec, R_f); Mat rotation_f = NVLib::PoseUtils::GetPose(R_f, Vec3d(0, 0, 0));

    logger.Log(1, "Finding pose");
    Mat translation_1 = NVLib::PoseUtils::CreateTPose(point_1);
    Mat translation_2 = NVLib::PoseUtils::CreateTPose(-point_2);
    
    Mat rotation = rotation_f * rotation_b;
    Mat pose = translation_1 * rotation * translation_2;
    
    cout << "Translation 1: " << endl << translation_1 << endl;
    cout << "Rotation: " << endl << rotation << endl;
    cout << "Translation 2: " << endl << translation_2 << endl;
    
    cout << "Pose: " << endl << pose << endl;

    cout << "Performing Test" << endl;
    cout << "Box Normal 1: " << boxNormal_1 << endl;
    cout << "Box Normal 2: " << boxNormal_2 << endl;
    cout << "Relative Rotation: " << endl << rotation << endl;

    Mat est_box_normal_1 = R_f * R_b * boxNormal_2;
    cout << "Estimated Box Normal 1: " << est_box_normal_1.t() << endl;

    logger.Log(1, "Applying pose to the model");
    Mat tcloud_2 = NVLib::CloudUtils::TransformCloud(cloud_2, pose);

    //Mat color_loc_1 = uColor_1.clone(); circle(color_loc_1, Point2d(323, 166), 5, Vec3b(0, 0, 255), 2);
    //Mat color_loc_2 = uColor_2.clone(); circle(color_loc_2, matchPoint, 5, Vec3b(0, 0, 255), 2);
    //imwrite("Output/color_loc_1.png", color_loc_1);
    //imwrite("Output/color_loc_2.png", color_loc_2);

    //logger.Log(1, "Rendering Planes");
    //SavePlane("Output/boxPlane_1.ply", boxPlane_1, Vec3d(0, 0, 255));
    //SavePlane("Output/boxPlane_2.ply", boxPlane_2, Vec3d(0, 0, 255));
    //SavePlane("Output/floorPlane_1.ply", floorPlane_1, Vec3d(0, 255, 0));
    //SavePlane("Output/floorPlane_2.ply", floorPlane_2, Vec3d(0, 255, 0));


    //NVLib::CloudUtils::Save("Output/cloud_1.ply", cloud_1);
    //NVLib::CloudUtils::Save("Output/cloud_2.ply", cloud_2);
    NVLib::CloudUtils::Save("Output/tcloud_2.ply", tcloud_2);
 
    //auto parts = vector<Mat>(); split(gradient_1, parts);
    //Mat gradientImage; parts[2].convertTo(gradientImage, CV_8U, 255.0);
    //imwrite("Output/gradient_1.png", gradientImage);
    //Mat gradientDisplay = GetGradientImage(gradient_1);
    //imwrite("Output/gradientImage_1.png", gradientDisplay);

    logger.Log(1, "Free Data");
    delete calibration;

    logger.StopApplication();
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
// Get Gradient
//--------------------------------------------------

/**
 * Get the gradient of the depth image
 * @param calibration The calibration object
 * @param cloud The depth image
 * @return The gradient of the depth image
 */
Mat GetGradient(NVLib::MonoCalibration * calibration, Mat& cloud) 
{
    Mat result = Mat_<Vec3d>::zeros(cloud.rows, cloud.cols);

    for (auto row = 1; row < result.rows - 1; row++) 
    {
        for (auto column = 1; column < result.cols - 1; column++) 
        {
            auto p = ReadPoint(cloud, row, column);
            if (p.z == 0) continue;

            auto p1 = ReadPoint(cloud, row - 1, column);
            auto p2 = ReadPoint(cloud, row + 1, column);
            auto p3 = ReadPoint(cloud, row, column - 1);
            auto p4 = ReadPoint(cloud, row, column + 1);
            if (p1.z == 0 || p2.z == 0 || p3.z == 0 || p4.z == 0) continue;

            auto v1 = p1 - p2; auto v2 = p3 - p4;
            auto v3 = Vec3d(v2.cross(v1));
            v3 = NVLib::Math3D::NormalizeVector(v3);

            result.at<Vec3d>(row, column) = v3;
        }
    }

    return result;
}

/**
 * Read a point from the cloud
 * @param cloud The cloud that we are reading from
 * @param row The row that we are reading from
 * @param column The column that we are reading from
 * @return The point that we are reading
 */
Point3d ReadPoint(Mat& cloud, int row, int column) 
{
    auto index = column + row * cloud.cols;
    auto data = (double*)cloud.data;

    auto X = data[index * 6 + 0];
    auto Y = data[index * 6 + 1];
    auto Z = data[index * 6 + 2];

    return Point3d(X, Y, Z);
}

/**
 * Get the gradient map
 * @param gradient The gradient image
 * @return The gradient map
 */
Mat GetGradientImage(Mat& gradient) 
{
    Mat result = Mat_<Vec3b>::zeros(gradient.size());

    for (auto row = 1; row < result.rows - 1; row++) 
    {
        for (auto column = 1; column < result.cols - 1; column++) 
        {
            auto g = gradient.at<Vec3d>(row, column);

            if (g[0] == 0 && g[1] == 0 && g[2] == 0) continue;

            if (g[0] > g[1] && g[0] > g[2]) 
            {
                result.at<Vec3b>(row, column) = Vec3b(255 * g[0], 0, 0);
            }
            else if (g[1] > g[0] && g[1] > g[2]) 
            {
                result.at<Vec3b>(row, column) = Vec3b(0, 255 * g[1], 0);
            }
            else if (g[2] > g[0] && g[2] > g[1]) 
            {
                result.at<Vec3b>(row, column) = Vec3b(0, 0, 255 * g[2]);
            }
        }
    }

    return result;
}

//--------------------------------------------------
// Part Extraction
//--------------------------------------------------

/**
 * Extract a part of the cloud
 * @param cloud The cloud that we are extracting from
 * @param gradient The gradient image
 * @param zMin The minimum Z value
 * @param zMax The maximum Z value
 * @param gradientMax The maximum gradient value
 * @param gThresh The gradient threshold
 */
Mat ExtractPart(Mat& cloud, Mat& gradient, double zMin, double zMax, int gradientMax, double gThresh, bool center) 
{
    Mat result = Mat_<uchar>::zeros(cloud.size());

    auto rowStart = center ? 250 : 0;
    auto rowEnd = center ? 320 : cloud.rows;
    auto columnStart = center ? 250 : 0;
    auto columnEnd = center ? 320 : cloud.cols;

    for (auto row = rowStart; row < rowEnd; row++) 
    {
        for (auto column = columnStart; column < columnEnd; column++) 
        {
            auto p = ReadPoint(cloud, row, column);
            if (p.z < zMin || p.z > zMax) continue;

            auto g = gradient.at<Vec3d>(row, column);
            if (gradientMax == 0 && (abs(g[0]) > gThresh && abs(g[0]) > abs(g[1]) && abs(g[0]) > abs(g[2]) == 0)) result.at<uchar>(row, column) = 255;
            else if (gradientMax == 1 && (abs(g[1]) > gThresh && abs(g[1]) > abs(g[0]) && abs(g[1]) > abs(g[2]))) result.at<uchar>(row, column) = 255;
            else if (gradientMax == 2 && (abs(g[2]) > gThresh && abs(g[2]) > abs(g[0]) && abs(g[2]) > abs(g[1]))) result.at<uchar>(row, column) = 255;
        }
    }

    return result;
}

//--------------------------------------------------
// Fit Plane Ransac
//--------------------------------------------------

/**
 * Fit a plane to the cloud using RANSAC
 * @param cloud The cloud that we are fitting to
 * @param mask The mask that we are using
 * @param iterations The number of iterations
 * @return The plane that we are fitting to
 */
Vec4d FitPlaneRansac(Mat& cloud, Mat& mask, int iterations) 
{
    auto points = vector<Point3d>();
    for (auto row = 0; row < cloud.rows; row++) 
    {
        for (auto column = 0; column < cloud.cols; column++) 
        {
            if (mask.at<uchar>(row, column) == 0) continue;

            auto p = ReadPoint(cloud, row, column);
            if (p.z == 0) continue;

            points.push_back(p);
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    auto sampleSize = (int)round(points.size() * 0.3); if (sampleSize < 10) throw runtime_error("Not enough points to sample from");
    auto bestScore = DBL_MAX; auto bestPlane = Vec4d(0, 0, 0, 0);

    for (auto i = 0; i < iterations; i++) 
    {
        auto pSample = vector<Point3d>(); 

        std::sample(points.begin(), points.end(), std::back_inserter(pSample), sampleSize, g);

        auto plane = NVLib::PlaneUtils::FitPlane(pSample);
        auto score = NVLib::PlaneUtils::AvePlaneError(plane, points);

        if (score[0] < bestScore) 
        {
            bestScore = score[0];
            bestPlane = plane;
        }
    }

    return bestPlane;
}

/**
 * Save the plane to a file
 * @param path The path to save the plane to
 * @param plane The plane that we are saving
 * @param color The color of the plane
 */
void SavePlane(const string& path, const Vec4d& plane, const Vec3d& color) 
{
    auto points = vector<Point3d>(); NVLib::PlaneUtils::BuildPlane(plane, points, Range(-500, 500), Range(-500, 500), 500);

    auto model = NVLib::Model();
    for (auto point : points) 
    {
        model.AddVertex(point, color);
    }

    NVLib::SaveUtils::SaveModel(path, &model);
}

//--------------------------------------------------
// Matching Point Logic
//--------------------------------------------------

/**
 * Defines the logic to find matching points
 * @param image_1 The first point that we are matching
 * @param image_1 The second point that we are matching
 * @param Point_1 The point that is being matched
 * @return The corresponding point
 */
Point2d FindMatch(Mat& image_1, Mat& image_2, const Point2d& point_1) 
{
    auto rect = Rect(point_1.x - 10, point_1.y - 10, 20, 20);
    auto kernel = image_1(rect);


    Mat scoreImage; matchTemplate(image_2, kernel, scoreImage, TM_CCOEFF_NORMED);
    double minVal, maxVal; Point minLoc, maxLoc;
    minMaxLoc(scoreImage, &minVal, &maxVal, &minLoc, &maxLoc);
    auto point_2 = Point2d(maxLoc.x + 10, maxLoc.y + 10);
    return point_2;
}

/**
 * Ray trace a point
 * @param calibration The calibration object
 * @param plane The plane that we are ray tracing
 * @param point The point that we are ray tracing
 * @return The point that we are ray tracing
 */
Point3d RayTrace(NVLib::MonoCalibration * calibration, const Vec4d& plane, const Point2d& point) 
{
    auto fx = calibration->GetCamera().at<double>(0, 0);
    auto fy = calibration->GetCamera().at<double>(1, 1);
    auto cx = calibration->GetCamera().at<double>(0, 2);
    auto cy = calibration->GetCamera().at<double>(1, 2);

    auto x = (point.x - cx) / fx;
    auto y = (point.y - cy) / fy;
    auto z = 1.0;

    auto ray = Vec3d(x, y, z);
    auto planeNormal = Vec3d(plane[0], plane[1], plane[2]);
    auto magnitude = sqrt(planeNormal[0] * planeNormal[0] + planeNormal[1] * planeNormal[1] + planeNormal[2] * planeNormal[2]);
    planeNormal[0] /= magnitude; planeNormal[1] /= magnitude; planeNormal[2] /= magnitude;
    auto d = plane[3] / magnitude;
    auto t = -d / (planeNormal[0] * ray[0] + planeNormal[1] * ray[1] + planeNormal[2] * ray[2]);
    auto point3d = Vec3d(ray[0] * t, ray[1] * t, ray[2] * t);

    return point3d;
}

//--------------------------------------------------
// Pose Logic
//--------------------------------------------------

/**
 * Find the rotation between two planes
 * @param boxPlane The box plane
 * @param floorPlane The floor plane
 * @return The rotation matrix
 */
Mat FindRotation(Vec4d& boxPlane, Vec4d& floorPlane) 
{
    auto boxNormal = Vec3d(boxPlane[0], boxPlane[1], boxPlane[2]);
    auto floorNormal = Vec3d(floorPlane[0], floorPlane[1], floorPlane[2]);

    auto axis_1 = NVLib::Math3D::NormalizeVector(boxNormal); 
    auto axis_2 = NVLib::Math3D::NormalizeVector(floorNormal.cross(boxNormal));
    auto axis_3 = NVLib::Math3D::NormalizeVector(axis_1.cross(axis_2));

    Mat rotation = Mat::eye(4, 4, CV_64F);
    rotation.at<double>(0, 0) = axis_1[0]; rotation.at<double>(0, 1) = axis_1[1]; rotation.at<double>(0, 2) = axis_1[2];
    rotation.at<double>(1, 0) = axis_2[0]; rotation.at<double>(1, 1) = axis_2[1]; rotation.at<double>(1, 2) = axis_2[2];
    rotation.at<double>(2, 0) = axis_3[0]; rotation.at<double>(2, 1) = axis_3[1]; rotation.at<double>(2, 2) = axis_3[2];

    cout << "OutRotation: " << endl << rotation << endl;

    return rotation;
}
