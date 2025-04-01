//--------------------------------------------------
// Implementation of class PoseMatchProblem
//
// @author: Wild Boar
//
// @date: 2024-08-10
//--------------------------------------------------

#include "PoseMatchProblem.h"
using namespace NVL_App;

//--------------------------------------------------
// Constructors and Terminators
//--------------------------------------------------

/**
 * @brief Custom Constructor
 * @param logger The logger that we are dealing with
 * @param camera The camera matrix associated with the system
 * @param frame_1 The first frame that we are evaluating
 * @param frame_2 The second frame that we are evaluating
 */
PoseMatchProblem::PoseMatchProblem(NVLib::Logger * logger, Mat& camera, MaskDepthFrame * frame_1, MaskDepthFrame * frame_2) : NVL_AI::ProblemBase(), _camera(camera), _logger(logger)
{
	_mask_1 = frame_1->GetMask().clone();
	_depth_1 = frame_1->GetMaskDepth().clone();

	_cloud_2 = NVL_App::HelperUtils::ExtractCloud(camera, frame_2);
    _normal_2 = NVL_App::HelperUtils::FindNormals(camera, frame_2);
}

//--------------------------------------------------
// Data Size
//--------------------------------------------------

/**
 * @brief The size of the training data set
 * @return int Returns a int
 */
int PoseMatchProblem::GetDataSize()
{
	return 6; // Because there are 6 unknowns
}

//--------------------------------------------------
// Evaluate
//--------------------------------------------------

/**
 * @brief Evaluate a particular solution against the training data
 * @param parameters The parameters that we are evaluating
 * @param errors The list of errors we got from the evaluation
 * @return double Returns a double
 */
double PoseMatchProblem::Evaluate(Mat& parameters, Mat& errors)
{
	// Reconstruct the pose
	auto plink = (double *) parameters.data;
	auto rvec = Vec3d(plink[0], plink[1], plink[2]); auto tvec = Vec3d(plink[3], plink[4], plink[5]);
	auto pose = (Mat)NVLib::PoseUtils::Vectors2Pose(rvec, tvec);

	// Get the score
	auto score = HelperUtils::EvalDiff(_camera, pose, _mask_1, _depth_1, _cloud_2, _normal_2);

	// Output the error to the console
	_logger->Log(1, "Error: %f", score);

	// Build the result
	auto elink = (double *) errors.data; for (auto i = 0; i < 6; i++) elink[i] = score;

	// Return the result
	return score;
}