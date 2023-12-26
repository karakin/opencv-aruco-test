#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <opencv2/core/utility.hpp>

int threshold_min_value = 0;
int threshold_value = 0;
int threshold_type = 3;

int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

bool showUi = false;


int main(int argc, char** argv)
{
    std::string keys =
        "{help h usage ? |      | print this message   }"
        "{gui            |      | show ui              }"
        "{id            |0     | video capture id     }";


    const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
    const char* trackbar_value = "Value";
    const char* trackbar_min_value = "Min Value";

    

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV test");
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    const int videoDeviceId = parser.get<int>("id");
    showUi = parser.has("gui");
    
    if (showUi)
    {
        cv::namedWindow("controls", cv::WINDOW_AUTOSIZE); // Create a window to display results

        cv::createTrackbar(trackbar_type,
            "controls", &threshold_type,
            max_type); 

        cv::createTrackbar(trackbar_value,
            "controls", &threshold_value,
            max_value);

        cv::createTrackbar(trackbar_min_value,
            "controls", &threshold_min_value,
            max_value);
    }

    const cv::Size2f transformedSize(320.f, 320.f);
    std::vector<cv::Point2f> targetPoints = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(transformedSize.width, 0.f) ,
        cv::Point2f(transformedSize.width, transformedSize.height),
        cv::Point2f(0, transformedSize.height)
    };
    cv::Mat transfromed = cv::Mat::zeros(transformedSize, CV_8UC3);

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
   
    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255) };
    
    cv::VideoCapture cap;

    cap.open(videoDeviceId, cv::CAP_ANY);
    if (!cap.isOpened())
    {
        std::cerr << "Unable to open camera\n";
        return EXIT_FAILURE;   
    }

    const std::string caption = cap.getBackendName();

    if(showUi)
    {
        cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);
        cv::namedWindow("transformed", cv::WINDOW_AUTOSIZE);
    }

    while (cap.grab())
    {
        cv::Mat image;
        cv::Mat gray;

        cv::Mat thresholded;

        auto start = std::chrono::high_resolution_clock::now();

        cap.retrieve(image);

        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, thresholded, threshold_min_value, threshold_value, threshold_type);

      

        detector.detectMarkers(thresholded, markerCorners, markerIds);
        
        for (int i = 0; i < markerIds.size(); ++i)
        {
            const std::vector<cv::Point2f>& corners = markerCorners[i];
            auto id = markerIds[i];

            cv::Mat transformationMatrix = cv::getPerspectiveTransform(corners, targetPoints);

            cv::warpPerspective(thresholded, transfromed, transformationMatrix, transfromed.size());

            if (showUi)
            {
                cv::imshow("transformed", transfromed);
            }
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "frame time " << duration.count() << " ms \n";

        if (showUi)
        {
            cv::imshow(caption, thresholded);

            if (cv::waitKey(5) >= 0)
                break;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

    }

    cap.release();

    return EXIT_SUCCESS;
}
