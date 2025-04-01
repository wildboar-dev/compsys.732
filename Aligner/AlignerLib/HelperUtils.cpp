//--------------------------------------------------
// Implementation of class HelperUtils
//
// @author: Wild Boar
//
// @date: 2024-07-30
//--------------------------------------------------

#include "HelperUtils.h"
using namespace NVL_App;

//--------------------------------------------------
// Overlap
//--------------------------------------------------

/**
 * @brief Find the overlap between two masks
 * @param mask_1 The first depth map that we are using
 * @param mask_2 The second depth map that we are using
 * @return Mat Returns a Mat
 */
Mat HelperUtils::MaskOverlap(Mat& mask_1, Mat& mask_2)
{
	assert(mask_1.cols == mask_2.cols && mask_1.rows == mask_2.rows);

	Mat result = Mat_<uchar>::zeros(mask_1.size());

	for (auto row = 0; row < result.rows; row++) 
	{
		for (auto column = 0; column < result.cols; column++) 
		{
			auto index = column + row * result.cols;

			auto c_1 = mask_1.data[index];
			auto c_2 = mask_2.data[index];

			auto value = c_1 > 0 && c_2 > 0 ? 255 : 0;

			result.data[index] = value;	
		}
	}

	return result;
}

//--------------------------------------------------
// Static Cloud
//--------------------------------------------------

/**
 * Build a static cloud application
 * @param camera The associated camera matrix
 * @param frame The frame that we are building the cloud from
 * @return The resultant cloud matrix
 */
Mat HelperUtils::ExtractCloud(Mat& camera, MaskDepthFrame * frame) 
{
	auto size = frame->GetDepth().size();
	auto result = (Mat) Mat_<Vec3d>::zeros(size); auto rlink = (double *) result.data;
	auto dlink = (double * ) frame->GetDepth().data;

	auto clink = (double *)camera.data;
	auto fx = clink[0]; auto fy = clink[4];
	auto cx = clink[2]; auto cy = clink[5];

	for (auto row = 0; row < size.height; row++) 
	{
		for (auto column = 0; column < size.width; column++) 
		{
			auto index = column + row * size.width;
			if (frame->GetMask().data[index] == 0) continue;

			auto Z = dlink[index]; 
			auto X = (column - cx) * (Z / fx);
			auto Y = (row - cy) * (Z / fy);
			if (Z <= 0) continue;

			rlink[index * 3 + 0] = X;
			rlink[index * 3 + 1] = Y;
			rlink[index * 3 + 2] = Z;
		}
	}

	return result;
}

/**
 * The main extract cloud for remapped point
 * @param camera The camera that we are dealing with
 * @param imagePoints The images points that we are dealing with
 * @param depth The depht that we are dealing with
 */
Mat HelperUtils::ExtractCloud(Mat& camera, Mat& imagePoints, Mat& depth) 
{
	auto result = (Mat)Mat_<Vec3d>::zeros(imagePoints.size());

	auto clink = (double *)camera.data;
	auto fx = clink[0]; auto fy = clink[4];
	auto cx = clink[2]; auto cy = clink[5];

	auto ilink = (float *) imagePoints.data;
	auto dlink = (double *) depth.data;
	auto output = (double *) result.data;

	for (auto row = 0; row < imagePoints.rows; row++) 
	{
		for (auto column = 0; column < imagePoints.cols; column++) 
		{
			auto index = column + row * imagePoints.cols;

			auto Z = dlink[index];
			if (Z == 0) continue;

			auto u = ilink[index * 2 + 0];
			auto v = ilink[index * 2 + 1];
			if (u == -1 && v == -1) continue;

			auto X = (u - cx) * (Z / fx);
			auto Y = (v - cy) * (Z / fy);
		
			output[index * 3 + 0] = X;
			output[index * 3 + 1] = Y;
			output[index * 3 + 2] = Z;
		}
	}

	return result;
}

/**
 * Save the given cloud to disk
 * @param path The path that we are saving to
 * @param color The color image associated with the cloud
 * @param cloud The cloud that we are saving
 */
void HelperUtils::SaveCloud(const string& path, Mat& color, Mat& cloud) 
{
	auto model = NVLib::Model();

	auto clink = (double * ) cloud.data;

	for (auto row = 0; row < cloud.rows; row++) 
	{
		for (auto column = 0; column < cloud.cols; column++) 
		{
			auto index = column + row * cloud.cols;

			auto X = clink[index * 3 + 0];
			auto Y = clink[index * 3 + 1];
			auto Z = clink[index * 3 + 2];

			auto B = color.data[index * 3 + 2];
			auto G = color.data[index * 3 + 1];
			auto R = color.data[index * 3 + 0];

			if (Z == 0) continue;

			auto colorPoint = NVLib::ColorPoint(X, Y, Z, R, G, B);
			model.AddVertex(colorPoint);			
		}
	}

	NVLib::SaveUtils::SaveModel(path, &model);
}

//--------------------------------------------------
// Extract Depth Map
//--------------------------------------------------

/**
 * Get the associated depth map
 * @param camera The given camera matrix
 * @param cloud The associated point cloud
 */
Mat HelperUtils::ExtractDepthMap(Mat& camera, Mat& cloud) 
{
	auto result = (Mat) Mat_<double>::zeros(cloud.size());
	auto dlink = (double *) result.data;

	auto clink = (double *) cloud.data;

	for (auto row = 0; row < result.rows; row++) 
	{
		for (auto column = 0; column < result.cols; column++) 
		{
			auto index = column + row * result.cols;

			auto X = clink[index * 3 + 0];
			auto Y = clink[index * 3 + 1];
			auto Z = clink[index * 3 + 2];
			if (Z == 0) continue;

			auto point = NVLib::Math3D::Project(camera, Point3d(X, Y, Z));

			auto u = (int)round(point.x); auto v = (int)round(point.y);
			if (u < 0 || u > result.cols || v < 0 || v > result.rows) continue;
			auto index2 = u + v * result.cols;

			dlink[index2] = Z;
		}
	}

	return result;
}

//--------------------------------------------------
// Find the normal vectors for a frame
//--------------------------------------------------

/**
 * Add the logic to find normals
 * @param frame The frame that we are finding normals for
 * @return A matrix containing the normal vectors
 */
Mat HelperUtils::FindNormals(Mat& camera, MaskDepthFrame * frame) 
{
	// Create a container to hold the result
	auto result = (Mat) Mat_<Vec3d>::zeros(frame->GetDepth().size());
	auto link = (double *) result.data;

	// Blur the depth map (fill holes and try to make more smooth)
	Mat fdepth; frame->GetDepth().convertTo(fdepth, CV_32FC1);
	Mat fresult; bilateralFilter(fdepth, fresult, 3, 3, 3);
	Mat depth; fresult.convertTo(depth, CV_64FC1);

	// Get cloud
	auto cframe = MaskDepthFrame(frame->GetColor(), depth, frame->GetMask());
	auto cloud = (Mat) ExtractCloud(camera, &cframe);

	// Loop thru and find normals for non-mask points
	for (auto row = 0; row < cloud.rows; row++) 
	{
		for (auto column = 0; column < cloud.cols; column++) 
		{
			auto index = column + row * cloud.cols;
			if (frame->GetMask().data[index] == 0) continue;

			auto points = vector<Point3d>(); GetNeighbourhood(cloud, Point(column, row), 14, points);
			if (points.size() < 4) continue;

			auto plane = NVLib::PlaneUtils::FitPlane(points);
			auto normal = Vec3d(plane[0], plane[1], plane[2]); normal = NVLib::Math3D::NormalizeVector(normal);

			link[index * 3 + 0] = normal[0];
			link[index * 3 + 1] = normal[1];
			link[index * 3 + 2] = normal[2];
		}
	}

	// Return result
	return result;
}

/**
 * Add the logic to get the associated points
 * @param cloud The cloud that we are getting points from
 * @param location The location of the point that we are extracting the neighbourhood for
 * @param blockSize The size of the neighbourhood that we are extracting from
 * @param points The resultant point set that we have extracted
 */
void HelperUtils::GetNeighbourhood(Mat& cloud, const Point& location, int blockSize, vector<Point3d>& points) 
{
	// Make sure that the output is clear
	points.clear();
	
	// Find the offset of the location from the block size (assume location is in the middle)
	auto offset = blockSize / 2;
	
	// Find the ranges
	auto xmin = max(0, location.x - offset); auto xmax = min(cloud.cols, location.x + offset + 1);
	auto ymin = max(0, location.y - offset); auto ymax = min(cloud.rows, location.y + offset + 1);

	// Create a link
	auto link = (double *) cloud.data;

	// Extract the points
	for (auto v = ymin; v < ymax; v++) 
	{
		for (auto u = xmin; u < xmax; u++) 
		{
			auto index = u + v * cloud.cols;
			auto X = link[index * 3 + 0];
			auto Y = link[index * 3 + 1];
			auto Z = link[index * 3 + 2];
			if (Z <= 0) continue;
			points.push_back(Point3d(X, Y, Z));
		}
	}
}

//--------------------------------------------------
// Extract Z Map
//--------------------------------------------------

/**
 * Determine a Z map
 * @param normals The normals map that we are working with
 */
Mat HelperUtils::ExtractZMap(Mat& normals) 
{
	auto parts = vector<Mat>(); split(normals, parts);
	Mat result; parts[2].convertTo(result, CV_8U, 255);
	return result;
}

//--------------------------------------------------
// Transformations
//--------------------------------------------------

/**
 * Add the functionality to transform a cloud
 * @param pose The pose that we are transforming
 * @param cloud The cloud that we are transforming
 * @return The transformed cloud (not positionally transformed)
 */
Mat HelperUtils::TransformCloud(Mat& pose, Mat& cloud) 
{
	Mat result = (Mat) Mat_<Vec3d>::zeros(cloud.size());

	auto input = (double *) cloud.data;
	auto output = (double *) result.data;

	for (auto row = 0; row < cloud.rows; row++) 
	{
		for (auto column = 0; column < cloud.cols; column++) 
		{
			auto index = column + row * cloud.cols;

			auto X = input[index * 3 + 0];
			auto Y = input[index * 3 + 1];
			auto Z = input[index * 3 + 2];
			if (Z == 0) continue;

			auto point = Point3d(X, Y, Z); auto tpoint = NVLib::Math3D::TransformPoint(pose, point);

			output[index * 3 + 0] = tpoint.x;
			output[index * 3 + 1] = tpoint.y;
			output[index * 3 + 2] = tpoint.z;
		}
	}

	return result;
} 

/**
 * Transform the normal map
 * @param pose The "full" pose that we are transforming by
 * @param normal The normal that we are transforming
 * @return The transformed normal map (not positionally transformed)
 */
Mat HelperUtils::TransformNormals(Mat& pose, Mat& normal) 
{
	Mat result = (Mat) Mat_<Vec3d>::zeros(normal.size());

	auto input = (double *) normal.data;
	auto output = (double *) result.data;

	auto rotation = (Mat) NVLib::PoseUtils::GetPoseRotation(pose);

	for (auto row = 0; row < normal.rows; row++) 
	{
		for (auto column = 0; column < normal.cols; column++) 
		{
			auto index = column + row * normal.cols;

			auto X = input[index * 3 + 0];
			auto Y = input[index * 3 + 1];
			auto Z = input[index * 3 + 2];
			if (Z == 0) continue;

			auto point = Point3d(X, Y, Z); auto tpoint = NVLib::Math3D::RotatePoint(rotation, point);

			output[index * 3 + 0] = tpoint.x;
			output[index * 3 + 1] = tpoint.y;
			output[index * 3 + 2] = tpoint.z;
		}
	}

	return result;
}

//--------------------------------------------------
// Image Points
//--------------------------------------------------

/**
 * Add the logic to find the associated image points
 * @param camera The camera matrix that we are using
 * @param cloud The cloud that we are processing
 */
Mat HelperUtils::GetPoints2D(Mat& camera, Mat& cloud) 
{
	Mat result = (Mat) Mat_<Vec2f>(cloud.size()); result.setTo(Vec2f(-1, -1));

	auto clink = (double *)camera.data;
	auto fx = clink[0]; auto fy = clink[4];
	auto cx = clink[2]; auto cy = clink[5];

	auto input = (double *) cloud.data;
	auto output = (float *) result.data;

	for (auto row = 0; row < cloud.rows; row++) 
	{
		for (auto column = 0; column < cloud.cols; column++) 
		{
			auto index = column + row * cloud.cols;

			auto X = input[index * 3 + 0];
			auto Y = input[index * 3 + 1];
			auto Z = input[index * 3 + 2];
			if (Z == 0) continue;

			auto u = fx * (X / Z) + cx;
			auto v = fy * (Y / Z) + cy;

			output[index * 2 + 0] = (float)u;
			output[index * 2 + 1] = (float)v;
		}	
	}

	return result;
}
