//--------------------------------------------------
// The helper that we are dealing with
//
// @author: Wild Boar
//
// @date: 2024-07-30
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <NVLib/PoseUtils.h>
#include <NVLib/Model/Model.h>
#include <NVLib/SaveUtils.h>
#include <NVLib/PlaneUtils.h>
#include <NVLib/Math3D.h>
#include <NVLib/DisplayUtils.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include "Calibration.h"
#include "MaskDepthFrame.h"

namespace NVL_App
{
	class HelperUtils
	{
	public:
		static Mat MaskOverlap(Mat& mask_1, Mat& mask_2);
		static Mat ExtractCloud(Mat& camera, MaskDepthFrame * frame);
		static Mat ExtractCloud(Mat& camera, Mat& imagePoints, Mat& depth);
		static Mat ExtractDepthMap(Mat& camera, Mat& cloud);
		static void SaveCloud(const string& path, Mat& color, Mat& cloud);
		static Mat FindNormals(Mat& camera, MaskDepthFrame * frame);
		static Mat ExtractZMap(Mat& normals);
		static Mat TransformCloud(Mat& pose, Mat& cloud);
		static Mat TransformNormals(Mat& pose, Mat& normal);
		static Mat GetPoints2D(Mat& camera, Mat& cloud);
	private:
		static void GetNeighbourhood(Mat& cloud, const Point& location, int blockSize, vector<Point3d>& points);
	};
}
