//--------------------------------------------------
// Implementation of class ICPEngine
//
// @author: Wild Boar
//
// @date: 2024-08-11
//--------------------------------------------------

#include "ICPEngine.h"
using namespace NVL_App;

//--------------------------------------------------
// Constructors and Terminators
//--------------------------------------------------

/**
 * @brief Initializer Constructor
 * @param camera Initialize variable <camera>
 * @param frame_1 Initialize variable <frame_1>
 * @param frame_2 Initialize variable <frame_2>
 * @param pose Initialize variable <pose>
 */
ICPEngine::ICPEngine(Mat& camera, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2, const Mat& pose)
{
	_camera = camera;
	_frame_1 = frame_1;
	_frame_2 = frame_2;
	_pose = pose;
}

//--------------------------------------------------
// Refine Logic
//--------------------------------------------------

/**
 * Add the main ICP refinement logic
 * @param maxIterations The maximum allowable amount of iterations
 */
double ICPEngine::Refine(int maxIterations) 
{
	auto score = 0.0;

	for (auto i = 0; i < maxIterations; i++) 
	{
		auto points = GetPoints();
		auto A = (Mat) GetA(points.get());
		auto B = (Mat) GetB(points.get());
		auto x = (Mat)Mat(); solve(A, B, x, DECOMP_SVD);

		auto xlink = (double *) x.data;

		Mat tMat = NVLib::PoseUtils::CreateTPose(Vec3d(xlink[3], xlink[4], xlink[5]));
		Mat rxMat = NVLib::PoseUtils::CreateRX(xlink[0]);
		Mat ryMat = NVLib::PoseUtils::CreateRY(xlink[1]);
		Mat rzMat = NVLib::PoseUtils::CreateRZ(xlink[2]);
		Mat update = tMat * rzMat * ryMat * rxMat;

		// Find score
		Mat actual = A * x; Mat difference; absdiff(actual, B, difference);
		auto mean = Scalar(); meanStdDev(difference, mean, noArray()); 
		score = mean[0];

		cout << "Score: " << score << endl;

		_pose =  _pose * update.inv();
	}

	return score;
}

//--------------------------------------------------
// Refine Logic
//--------------------------------------------------

/**
 * Get the set of corresponding points
 * @return The set of corresponding points
 */
unique_ptr<MatchPoints> ICPEngine::GetPoints() 
{
	auto result = new MatchPoints();

	Mat sourceCloud = HelperUtils::ExtractCloud(_camera, _frame_1);
	Mat normals = HelperUtils::FindNormals(_camera, _frame_1);
	Mat destCloud = HelperUtils::ExtractCloud(_camera, _frame_2);
	Mat tdestCloud = HelperUtils::TransformCloud(_pose, destCloud);
	auto imagePoints = HelperUtils::GetPoints2D(_camera, tdestCloud);

	Mat display_1 = HelperUtils::ExtractDepthMap(_camera, sourceCloud);
	Mat display_2 = HelperUtils::ExtractDepthMap(_camera, tdestCloud);
	NVLib::DisplayUtils::ShowFloatMap("Map 1", display_1, 640);
	NVLib::DisplayUtils::ShowFloatMap("Map 2", display_2, 640);
	waitKey(30);

	auto s = (double *) sourceCloud.data;
	auto n = (double *) normals.data;
	auto d = (double *) tdestCloud.data;
	auto plink = (float *) imagePoints.data; 

	for (auto row = 0; row < destCloud.rows; row++) 
	{
		for (auto column = 0; column < destCloud.cols; column++) 
		{
			auto index_1 = column + row * destCloud.cols;

			auto dx = d[index_1 * 3 + 0];
			auto dy = d[index_1 * 3 + 1];
			auto dz = d[index_1 * 3 + 2];
			if (dz == 0) continue;

			auto ix = plink[index_1 * 2 + 0];
			auto iy = plink[index_1 * 2 + 1];
			auto u = (int)round(ix); auto v = (int)round(iy);
			if (u < 0 || u >= destCloud.cols || v < 0 || v >= destCloud.rows) continue;
			auto index_2 = u + v * destCloud.cols;

			auto nx = n[index_2 * 3 + 0];
			auto ny = n[index_2 * 3 + 1];
			auto nz = n[index_2 * 3 + 2];
			if (nx == 0 && ny == 0 && nz == 0) continue;

			auto sx = s[index_2 * 3 + 0];
			auto sy = s[index_2 * 3 + 1];
			auto sz = s[index_2 * 3 + 2];
			if (sz == 0) continue;

			auto delta = Vec3d(sx - dx, sy - dy, dz - dz);
			auto magnitude = NVLib::Math3D::GetMagnitude(delta);
			auto dotMag = delta.dot(Vec3d(nx, ny, nz));
			auto ratio = abs(dotMag / magnitude);
			if (ratio < 0.5 || magnitude > 0.002) continue;

			result->Add(Point3d(sx, sy, sz), Point3d(dx, dy, dz), Point3d(nx, ny, nz));
		}
	}

	return unique_ptr<MatchPoints>(result);
}

//--------------------------------------------------
// Matrix Construction
//--------------------------------------------------

/**
 * Add the logic to construct the given matrix
 * @param points The points that we are constructing
 * @return The resultant matrix
 */
Mat ICPEngine::GetA(MatchPoints * points) 
{
	Mat result = Mat_<double>(points->GetCount(), 6);
	auto link = (double *) result.data;

	for (auto i = 0; i < result.rows; i++) 
	{
		link[i * 6 + 0] = points->n(i).z * points->s(i).y - points->n(i).y * points->s(i).z;
		link[i * 6 + 1] = points->n(i).x * points->s(i).z - points->n(i).z * points->s(i).x;
		link[i * 6 + 2] = points->n(i).y * points->s(i).x - points->n(i).x * points->s(i).y;
		link[i * 6 + 3] = points->n(i).x;
		link[i * 6 + 4] = points->n(i).y;
		link[i * 6 + 5] = points->n(i).z;
	}

	return result;
}

/**
 * Retrieve the associated B points
 * @param points The points that we are retrieving
 * @return The result Matrix
 */
Mat ICPEngine::GetB(MatchPoints * points) 
{
	Mat result = Mat_<double>(points->GetCount(), 1);
	auto link = (double *) result.data;

	for (auto i = 0; i < result.rows; i++) 
	{
		auto v = (points->n(i).x * points->d(i).x * points->n(i).y * points->d(i).y + points->n(i).z * points->d(i).z)
				 - (points->n(i).x * points->s(i).x * points->n(i).y * points->s(i).y + points->n(i).z * points->s(i).z);
		link[i] = v;
	}

	return result;
}