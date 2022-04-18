#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void scaleFaces(vector<cv::Rect>& faces) {
    for (auto& face : faces)
    {
        float percent = 0.2f;
        cv::Size deltaSize(face.width * percent, face.height * percent);
        cv::Point offset(deltaSize.width / 2, deltaSize.height / 2);
        face += deltaSize;
        face -= offset;
    }
}