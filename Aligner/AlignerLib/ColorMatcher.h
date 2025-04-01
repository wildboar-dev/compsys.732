//--------------------------------------------------
// The color aligner that we are dealing with
//
// @author: Wild Boar
//
// @date: 2024-08-15
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
using namespace cv;

#include "MaskDepthFrame.h"

namespace NVL_App
{
	class ColorMatcher
	{
	private:
		vector<int> _iterCounts;
		vector<float> _minGradMagnitudes;

		const float minDepth = 0.1f;
    	const float maxDepth = 0.3f; 
    	const float maxDepthDiff = 0.01f; 
	public:
		ColorMatcher();

		Mat FindPose(Mat camera, Mat initPose, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2);

		inline vector<int>& GetIterCounts() { return _iterCounts; }
		inline vector<float>& GetMinGradMagnitudes() { return _minGradMagnitudes; }
	};
}
