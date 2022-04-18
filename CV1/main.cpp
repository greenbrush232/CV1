#include <iostream>
#include <opencv2/opencv.hpp>

#include"scaleFace.h"

using namespace std;

void main() {
    string image_path = "Lenna.png";
    string cascade_path = "haarcascade_frontalface_alt2.xml";

    cv::Mat source_image = cv::imread(image_path);
    cv::imshow("Image", source_image);
    cv::CascadeClassifier face_cascade;
    face_cascade.load(cascade_path);

    // Detecting faces
    vector<cv::Rect> faces;
    face_cascade.detectMultiScale(source_image, faces, 1.1, 4);

    // Scale
    scaleFaces(faces);
    cv::Mat rectangles = source_image.clone();
    for (auto& face : faces)
        cv::rectangle(rectangles, face.tl(), face.br(), cv::Scalar(255, 0, 0), 2);
    cv::imshow("Rectangles", rectangles);

    cv::Mat face_image = source_image(faces[0]);
    cv::imshow("Face", face_image);

    // Find contours
    vector<vector<cv::Point>> contPoints;
    cv::Mat contours = cv::Mat::zeros(face_image.size(), CV_8UC3);
    cv::Mat canny;

    cv::Canny(face_image, canny, 100, 200);
    cv::findContours(canny, contPoints, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(contours, contPoints, -1, cv::Scalar(255, 255, 255), 1);

    cv::imshow("Contours1", contours);

    // Erase contours
    cv::Mat newContours = cv::Mat::zeros(face_image.size(), CV_8UC3);

    contPoints.erase(remove_if(contPoints.begin(), contPoints.end(),
        [](vector<cv::Point> const& x) {
            return cv::arcLength(x, false) <= 10;
        }), contPoints.end());
    cv::drawContours(newContours, contPoints, -1, cv::Scalar(255, 255, 255), 1);

    cv::imshow("Contours2", newContours);

    // Dilation
    cv::Mat dilated;

    cv::dilate(newContours, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::imshow("Dilation", dilated);

    // Gauss
    cv::Mat gaussian, normalised;

    cv::GaussianBlur(dilated, gaussian, cv::Size(5, 5), 3);
    cv::normalize(gaussian, normalised, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
    cv::imshow("Gaussian", normalised);

    // Bilateral
    cv::Mat f1;

    cv::bilateralFilter(face_image, f1, 15, 80, 80);
    cv::imshow("Bilateral", f1);

    // F2
    double sigma = 1, amount = 3;
    cv::Mat f2, blur;

    cv::GaussianBlur(face_image, blur, cv::Size(), sigma);
    cv::addWeighted(face_image, 1 + amount, blur, -amount, 0, f2);
    cv::imshow("Sharp", f2);

    // Filtration
    cv::Mat res = cv::Mat::zeros(face_image.size(), CV_8UC3);

    for (int x = 0; x < face_image.cols; x++)
    {
        for (int y = 0; y < face_image.rows; y++)
        {
            cv::Vec3b res_pixel;
            cv::Vec3b f1_pixel = f1.at<cv::Vec3b>(x, y);
            cv::Vec3b f2_pixel = f2.at<cv::Vec3b>(x, y);
            float m_pixel = normalised.at<float>(x, y);

            for (int c = 0; c < 3; c++)
                res_pixel[c] = m_pixel * f2_pixel[c] + (1.0 - m_pixel) * f1_pixel[c];

            res.at<cv::Vec3b>(cv::Point(y, x)) = res_pixel;
        }
    }

    cv::imshow("Result", res);

    cv::waitKey(0);
    system("pause");
};