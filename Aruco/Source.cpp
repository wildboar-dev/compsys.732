//--------------------------------------------------
// Startup code module
//
// @author: Wild Boar
//
// @date: 2025-03-30
//--------------------------------------------------

#include <iostream>
using namespace std;

#include <NVLib/Logger.h>
#include <NVLib/Path/PathHelper.h>
#include <NVLib/LoadUtils.h>
#include <NVLib/Parameters/Parameters.h>

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "ArgReader.h"

//--------------------------------------------------
// Function Prototypes
//--------------------------------------------------
void Run(NVLib::Parameters * parameters);

//--------------------------------------------------
// Execution Logic
//--------------------------------------------------

/**
 * Main entry point into the application
 * @param parameters The input parameters
 */
void Run(NVLib::Parameters * parameters) 
{
    if (parameters == nullptr) return; auto logger = NVLib::Logger(1);

    logger.StartApplication();
    cv::VideoCapture inputVideo;
    inputVideo.open(2);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    auto calibration = NVLib::LoadUtils::LoadCalibration("calibration.xml");

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);

        // if at least one marker detected
        if (ids.size() > 0) 
        {
            Mat camera = calibration->GetCamera();
            Mat distortion = calibration->GetDistortion();

            std::vector<cv::Vec3d> rvecs, tvecs;cv::aruco::estimatePoseSingleMarkers(corners, 0.036, camera, distortion, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

            for (int i = 0; i < rvecs.size(); ++i) 
            {
                auto rvec = rvecs[i];
                auto tvec = tvecs[i];
                cv::drawFrameAxes(imageCopy, camera, distortion, rvec, tvec, 0.1);
            }

        }

        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(30);
        if (key == 27)
            break;
    }

    delete calibration;

    logger.StopApplication();
}

//--------------------------------------------------
// Entry Point
//--------------------------------------------------

/**
 * Main Method
 * @param argc The count of the incoming arguments
 * @param argv The number of incoming arguments
 * @return SUCCESS and FAILURE
 */
int main(int argc, char ** argv) 
{
    NVLib::Parameters * parameters = nullptr;

    try
    {
        parameters = NVL_Utils::ArgReader::GetParameters(argc, argv);
        Run(parameters);
    }
    catch (runtime_error exception)
    {
        cerr << "Error: " << exception.what() << endl;
        exit(EXIT_FAILURE);
    }
    catch (string exception)
    {
        cerr << "Error: " << exception << endl;
        exit(EXIT_FAILURE);
    }

    if (parameters != nullptr) delete parameters;

    return EXIT_SUCCESS;
}
