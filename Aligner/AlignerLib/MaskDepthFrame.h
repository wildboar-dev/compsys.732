//--------------------------------------------------
// The frame that we are processing
//
// @author: Wild Boar
//
// @date: 2024-07-29
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

namespace NVL_App
{
	class MaskDepthFrame
	{
	private:
		Mat _color;
		Mat _depth;
		Mat _mask;
	public:
		MaskDepthFrame(Mat& color, Mat& depth, Mat& mask) :
			_color(color), _depth(depth), _mask(mask) {}

		Mat GetMaskDepth() { Mat result; _depth.copyTo(result, _mask); return result; }

		inline Mat& GetColor() { return _color; }
		inline Mat& GetDepth() { return _depth; }
		inline Mat& GetMask() { return _mask; }
	};
}
