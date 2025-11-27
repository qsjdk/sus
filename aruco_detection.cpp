#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main()
{
    // 定义ArUco字典和参数
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    // 相机标定参数
    double fx = 406.932130;
    double fy = 402.678201;
    double cx = 316.629381;
    double cy = 242.533947;

    double k1 = 0.039106;
    double k2 = -0.056494;
    double p1 = -0.000824;
    double p2 = 0.092161;
    double k3 = 0.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);


    // 打开相机
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "无法打开相机" << std::endl;
        return -1;
    }

    while (true)
    {
        // 读取相机帧
        cv::Mat frame;
        cap >> frame;

        // 转换为灰度图像
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 检测ArUco二维码
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        // 绘制检测结果
        if (!markerIds.empty())
        {
            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

            for (size_t i = 0; i < markerIds.size(); ++i)
                {
                    cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

                    std::cout << "Marker ID: " << markerIds[i] << std::endl;
                    std::cout << "  tvec (x, y, z): "
                              << tvecs[i][0] << ", "
                              << tvecs[i][1] << ", "
                              << tvecs[i][2] << std::endl;
                    std::cout << "  rvec (rx, ry, rz): "
                              << rvecs[i][0] << ", "
                              << rvecs[i][1] << ", "
                              << rvecs[i][2] << std::endl;
                }
        }


        // 显示图像
        cv::imshow("ArUco Detection", frame);

        // 按下 'q' 键退出循环
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
