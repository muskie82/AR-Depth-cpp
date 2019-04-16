#pragma once

#include <Eigen/Geometry>
#include <map>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const int width_resize = 180;
const int height_resize = 320;

class Point {
public:
    Point() = default;
    int id = -1;
    Eigen::Vector3d position3d = Eigen::Vector3d::Zero();
};

class Camera {
public:
    Camera() = default;
    int id = -1;
    int width = 0;
    int height = 0;
    float fx=0, fy=0;
    float cx=0, cy=0;
    std::string model="";

    void Print();
};

class View {
public:
    int id=-1;
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::map<int, std::pair<double, double>> points2d;
    int camera_id = -1;
    std::string name="";

    inline bool isKeyframe(){return (points2d.size()>0);
    }

    inline Eigen::Matrix3d Rotation(){ return orientation.normalized().toRotationMatrix();}

    inline Eigen::Vector3d Position(){ return orientation*translation;}

    cv::Mat GetImage(std::string image_folder, bool resize);

    void Print();
};

class Reconstruction {
public:
    std::map<int, Camera> cameras;
    std::map<int, View> views;
    std::map<int, Point> points3d;
    int min_view_id=-1;
    int max_view_id=-1;
    std::string image_folder="";

    std::vector<int> ViewIds();

    std::pair<int, int> GetNeighboringKeyframes(int view_id);
    std::vector<int> GetReferenceFrames(int view_id);

    cv::Mat GetImage(int view_id, bool resize);

    Eigen::Matrix3d GetQuat(int view_id);

    cv::Mat GetSparseDepthMap(int frame_id, bool resize);

};
