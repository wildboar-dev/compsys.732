//--------------------------------------------------
// The ICP engine
//
// @author: Wild Boar
//
// @date: 2024-08-11
//--------------------------------------------------

#pragma once

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <NVLib/Math3D.h>
#include <NVLib/PoseUtils.h>
#include <NVLib/DisplayUtils.h>

#include "MaskDepthFrame.h"
#include "HelperUtils.h"

namespace NVL_App
{
	class MatchPoints 
	{
	private:
		vector<Point3d> _s;
		vector<Point3d> _d;
		vector<Point3d> _n;
	public:
		inline void Add(const Point3d& s, const Point3d& d, const Point3d& n) 
		{
			_s.push_back(s); _d.push_back(d); _n.push_back(n);
		}

		inline int GetCount() { return _s.size(); }

		inline Point3d& s(int index) { return _s[index]; }
		inline Point3d& d(int index) { return _d[index]; }
		inline Point3d& n(int index) { return _n[index]; }
	};

	class ICPEngine
	{
	private:
		Mat _camera;
		MaskDepthFrame * _frame_1;
		MaskDepthFrame * _frame_2;
		Mat _pose;
	public:
		ICPEngine(Mat& camera, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2, const Mat& pose = Mat_<double>::eye(4,4));

		double Refine(int maxIterations);

		inline Mat& GetPose() { return _pose; }
	private:
		unique_ptr<MatchPoints> GetPoints();
		Mat GetA(MatchPoints * points);
		Mat GetB(MatchPoints * points);
	};
}
