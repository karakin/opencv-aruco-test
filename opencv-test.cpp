#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

int main(int argc, char** argv)
{

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
        
    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255) };
    
    cv::VideoCapture cap;

    cap.open(0, cv::CAP_ANY);
    if (!cap.isOpened())
    {
        std::cerr << "Unable to open camera\n";
        return -1;   
    }
    const std::string caption = cap.getBackendName();

    cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);

    while (cap.grab())
    {
        cv::Mat image;
        cap.retrieve(image);

        auto start = std::chrono::high_resolution_clock::now();
        detector.detectMarkers(image, markerCorners, markerIds, rejected);
        
        cv::Mat output = image.clone();

        for (int i = 0; i < markerIds.size(); ++i)
        {
            const std::vector<cv::Point2f>& corners = markerCorners[i];
            const cv::Point2f cX = (corners[0] + corners[2]) / 2;
            const cv::Point2f cY = (corners[1] + corners[3]) / 2;

            for (int c = 0; c < 4; ++c)
            {
                cv::circle(output, corners[c], 5, colors[c], 5);
            }
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "frame time " << duration.count() << " ms \n";

        cv::imshow(caption, output);

        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}
