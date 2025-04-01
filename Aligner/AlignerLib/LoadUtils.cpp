//--------------------------------------------------
// Implementation of class LoadUtils
//
// @author: Wild Boar
//
// @date: 2024-07-29
//--------------------------------------------------

#include "LoadUtils.h"
using namespace NVL_App;


//--------------------------------------------------
// Calibration
//--------------------------------------------------

/**
 * @brief Load the calibration
 * @param pathHelper The helper for formulating the correct pathHelper
 * @return unique_ptr<Calibration> Returns a unique_ptr<Calibration>
 */
unique_ptr<Calibration> LoadUtils::LoadCalibration(NVLib::PathHelper& pathHelper)
{
	auto path = pathHelper.GetPath("Meta", "calibration.xml"); 
	auto reader = FileStorage(path, FileStorage::FORMAT_XML | FileStorage::READ);
	if (!reader.isOpened()) throw runtime_error("Unable to open: " + path);
	Mat camera; reader["camera"] >> camera;

	Mat distortion; reader["distortion"] >> distortion;
	auto imageSize = Size(); reader["image_size"] >> imageSize;
	reader.release();
	return unique_ptr<Calibration>(new Calibration(camera, distortion, imageSize));
}

//--------------------------------------------------
// Frame
//--------------------------------------------------

/**
 * @brief Load a depth frame
 * @param pathHelper A helper for formulating the correct pathHelper
 * @param index The index of the frame that we are loading
 * @return unique_ptr<MaskDepthFrame> Returns a unique_ptr<MaskDepthFrame>
 */
unique_ptr<MaskDepthFrame> LoadUtils::LoadMaskFrame(NVLib::PathHelper& pathHelper, int index)
{
	// Define the file names
	auto colorFile = stringstream(); colorFile << "color_" << setw(4) << setfill('0') << index << ".png";
	auto depthFile = stringstream(); depthFile << "depth_" << setw(4) << setfill('0') << index << ".png";
	//auto maskFile = stringstream(); maskFile << "Mask_" << setw(4) << setfill('0') << index << ".png";

	// Get the image paths
	auto colorPath = pathHelper.GetPath("Frames", colorFile.str());
	auto depthPath = pathHelper.GetPath("Frames", depthFile.str());
	//auto maskPath = pathHelper.GetPath("Masks", maskFile.str());

	// Load the images
	auto color = (Mat) imread(colorPath); if (color.empty()) throw runtime_error("Unable to load: " + colorPath);
	auto depth = (Mat) imread(depthPath, IMREAD_UNCHANGED); if (depth.empty()) throw runtime_error("Unable to load: " + depthPath);
	auto mask = Mat_<uchar>(color.size()); mask.setTo(255); //(Mat) imread(maskPath, IMREAD_GRAYSCALE); if (mask.empty()) throw runtime_error("Unable to load: " + maskPath);
	auto floatDepth = (Mat) Mat(); depth.convertTo(floatDepth, CV_32FC1);
	//Mat filteredDepth; bilateralFilter(floatDepth, filteredDepth, 5, 60, 60);
	//Mat finalDepth; filteredDepth.convertTo(finalDepth, CV_64FC1, 1e-3);
	Mat finalDepth; floatDepth.convertTo(finalDepth, CV_64FC1, 1e-3);

	auto link = (double *) finalDepth.data;
	auto pixelCount = floatDepth.cols * floatDepth.rows;
	for (auto i = 0; i < pixelCount; i++)
	{
		auto Z = link[i]; 				
		//if (Z < 0.1 || Z > 8) link[i] = 0;
	}

	// Return the result
	return unique_ptr<MaskDepthFrame>(new MaskDepthFrame(color, finalDepth, mask));
}