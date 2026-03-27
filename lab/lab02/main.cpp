
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = samples::findFile("test_zmjjkk.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    std::cout << "Image size: " << img.rows << " x " << img.cols << std::endl;
    std::cout << "Number of channels: " << img.channels() << std::endl;
    std::cout << "Data type: " << img.type() << std::endl;

    Vec3b pixel = img.at<Vec3b>(0, 0);
    std::cout << "Pixel at (0,0): B=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", R=" << (int)pixel[2] << std::endl;

    Rect roi(350, 100, 300, 300);
    Mat cropped = img(roi);
    imwrite("cropped.jpg", cropped);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    imshow("Original Image", img);
    imshow("Gray Image", gray);
    imshow("cropped Image", cropped);
    imwrite("gray.jpg", gray);
    waitKey(0);

    return 0;
}