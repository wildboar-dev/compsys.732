//--------------------------------------------------
// Defines a basic engine for a vanilla C++ project.
//
// @author: Wild Boar
//
// @date: 2024-07-29
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <AlignerLib/ColorMatcher.h>

#include <NVLib/Logger.h>
#include <NVLib/DisplayUtils.h>
#include <NVLib/Path/PathHelper.h>

#include <JarvisLib/Solver/LMFinder.h>

#include <AlignerLib/ArgUtils.h>
#include <AlignerLib/MaskDepthFrame.h>
#include <AlignerLib/LoadUtils.h>
#include <AlignerLib/HelperUtils.h>
#include <AlignerLib/ICPEngine.h>

namespace NVL_App
{
	class Engine
	{
	private:
		NVLib::Parameters * _parameters;
		NVLib::Logger* _logger;
		NVLib::PathHelper * _pathHelper;
		int _count;
	public:
		Engine(NVLib::Logger* logger, NVLib::Parameters * parameters);
		~Engine();

		void Run();
	private:
		Mat EstimatePose(Mat& camera, int frameId, Mat& delta, double& score);
		void RenderModel(NVLib::PathHelper * pathHelper, Mat& camera, int frameId, Mat& pose);
		Mat ConvertPose(Mat& parameters);
	};
}