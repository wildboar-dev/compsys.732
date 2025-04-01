//--------------------------------------------------
// Load models from disk
//
// @author: Wild Boar
//
// @date: 2024-07-29
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <NVLib/DisplayUtils.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <NVLib/Path/PathHelper.h>
#include "Calibration.h"
#include "MaskDepthFrame.h"

namespace NVL_App
{
	class LoadUtils
	{
	public:
		static unique_ptr<Calibration> LoadCalibration(NVLib::PathHelper& path);
		static unique_ptr<MaskDepthFrame> LoadMaskFrame(NVLib::PathHelper& path, int index);
	};
}
