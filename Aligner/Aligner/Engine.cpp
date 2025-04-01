//--------------------------------------------------
// Implementation code for the Engine
//
// @author: Wild Boar
//
// @date: 2024-07-29
//--------------------------------------------------

#include "Engine.h"
using namespace NVL_App;

//--------------------------------------------------
// Constructor and Terminator
//--------------------------------------------------

/**
 * Main Constructor
 * @param logger The logger that we are using for the system
 * @param parameters The input parameters
 */
Engine::Engine(NVLib::Logger* logger, NVLib::Parameters* parameters) 
{
    _logger = logger; _parameters = parameters;

    _logger->Log(1, "Creating a path helper");
    auto database = ArgUtils::GetString(parameters, "database");
    auto dataset = ArgUtils::GetString(parameters, "dataset");
    _pathHelper = new NVLib::PathHelper(database, dataset);

    _logger->Log(1, "Loading the frame count");
    _count = ArgUtils::GetInteger(parameters, "count");
    _logger->Log(1, "Frame count set to: %i", _count);
}

/**
 * Main Terminator 
 */
Engine::~Engine() 
{
    delete _parameters; delete _pathHelper;
}

//--------------------------------------------------
// Execution Entry Point
//--------------------------------------------------

/**
 * Entry point function
 */
void Engine::Run()
{
    _logger->Log(1, "Loading the calibration object");
    auto calibration = NVL_App::LoadUtils::LoadCalibration(*_pathHelper);
    _logger->Log(1, "Image Size: [%i,%i]", calibration->GetImageSize().width, calibration->GetImageSize().height);

    _logger->Log(1, "Creating a global pose variable");
    auto globalPose = (Mat) Mat_<double>::eye(4,4);

    _logger->Log(1, "Writing the first frame to disk");
    RenderModel(_pathHelper, calibration->GetCamera(), 0, globalPose);

    Mat delta; auto score = DBL_MAX;

    for (auto i = 0; i < _count; i++) 
    {
        auto currentScore = 0.0;
        auto pose = EstimatePose(calibration->GetCamera(), i, delta, currentScore);
        
        if (currentScore < score) { delta = pose; score = currentScore; }

        globalPose *= pose;

        RenderModel(_pathHelper, calibration->GetCamera(), i+1, globalPose);
    }
}

//--------------------------------------------------
// Pose Estimation Logic
//--------------------------------------------------

/**
 * Add the logic to estimate pose
 * @param frameId The identifier of the "lower frame"
 * @param delta The estimated rotation delta
 * @param score The score that we are getting from
 * @return the pose estimate
 */
Mat Engine::EstimatePose(Mat& camera, int frameId, Mat& delta, double& score) 
{
    _logger->Log(1, "Loading the first frame: %i", frameId);
    auto frame_1 = NVL_App::LoadUtils::LoadMaskFrame(*_pathHelper, frameId);

    _logger->Log(1, "Loading the second frame: %i", frameId + 1);
    auto frame_2 = NVL_App::LoadUtils::LoadMaskFrame(*_pathHelper, frameId + 1);

    _logger->Log(1, "Creating an alignment engine");
    Mat guess = delta; if (guess.empty()) guess = Mat_<double>::eye(4,4);
    auto aligner = ICPEngine(camera, frame_1.get(), frame_2.get(), guess);
    
    _logger->Log(1, "Performing Alignment");
    score = aligner.Refine(50); Mat pose = aligner.GetPose();
    cout << "Pose: " << pose << " with score " << score << endl;

    _logger->Log(1, "Attempting color alignment");
    auto colorAligner = ColorMatcher(); Mat cPose = colorAligner.FindPose(camera, pose, frame_1.get(), frame_2.get());
    cout << "Color Pose: " << cPose << endl;

    return cPose;
}

//--------------------------------------------------
// Helper
//--------------------------------------------------

/**
 * Add the logic to render the assocated model to disk
 * @param path The helper to generate the associated path
 * @param camera The associated camera matrix
 * @param frameId The frame identifier
 * @param pose The pose that we are using
 */
void Engine::RenderModel(NVLib::PathHelper * pathHelper, Mat& camera, int frameId, Mat& pose) 
{
    _logger->Log(1, "Loading frame: %i", frameId);
    auto frame = NVL_App::LoadUtils::LoadMaskFrame(*_pathHelper, frameId);

    _logger->Log(1, "Generating save file name and path");
    auto fileName = stringstream(); fileName << "frame_" << setw(4) << setfill('0') << frameId << ".ply";
    auto path = pathHelper->GetPath("Models", fileName.str());

    _logger->Log(1, "Writing frame cloud to disk");
    Mat cloud = HelperUtils::ExtractCloud(camera, frame.get());
    Mat tcloud = HelperUtils::TransformCloud(pose, cloud);
    HelperUtils::SaveCloud(path, frame->GetColor(), tcloud);
}

//--------------------------------------------------
// Helper
//--------------------------------------------------

/**
 * Get the pose from the parameters
 * @param parameters The parameters that we are dealing with
 */
Mat Engine::ConvertPose(Mat& parameters) 
{
	// Reconstruct the pose
	auto plink = (double *) parameters.data;
	auto rvec = Vec3d(plink[0], plink[1], plink[2]); auto tvec = Vec3d(plink[3], plink[4], plink[5]);
	auto pose = (Mat)NVLib::PoseUtils::Vectors2Pose(rvec, tvec);

    // Return the result
    return pose;
}
