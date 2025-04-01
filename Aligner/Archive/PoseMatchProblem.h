//--------------------------------------------------
// Defines optimization problem of finding the best pose between RGBD frames
//
// @author: Wild Boar
//
// @date: 2024-08-10
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <NVLib/PoseUtils.h>
#include <NVLib/Logger.h>

#include <JarvisLib/Solver/ProblemBase.h>

#include "MaskDepthFrame.h"
#include "HelperUtils.h"

namespace NVL_App
{
	class PoseMatchProblem : public NVL_AI::ProblemBase
	{
	private:
		Mat _camera;
		Mat _mask_1;
		Mat _depth_1;
		Mat _cloud_2;
		Mat _normal_2;
		NVLib::Logger * _logger;
	public:
		PoseMatchProblem(NVLib::Logger * logger, Mat& camera, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2);

		virtual int GetDataSize();
		virtual double Evaluate(Mat& parameters, Mat& errors);

		inline Mat& GetCamera() { return _camera; }
		inline Mat& GetMask_1() { return _mask_1; }
		inline Mat& GetDepth_1() { return _depth_1; }
		inline Mat& GetCloud_2() { return _cloud_2; }
		inline Mat& GetNormal_2() { return _normal_2; }
	};
}
