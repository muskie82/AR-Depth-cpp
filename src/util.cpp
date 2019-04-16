#include "util.h"

void Camera::Print(){
    std::cout<<"Camera "<<id<<std::endl;
    std::cout<<"-Image size: ("<<width<<", "<<height<<")"<<std::endl;
    std::cout<<"-Focal "<<fx<<" "<<fy<<std::endl;
    std::cout<<"-Model "<<model<<std::endl;
}

cv::Mat View::GetImage(std::string image_folder, bool resize){
    cv::Mat img = cv::imread(image_folder+"/"+name);
    assert(img.data);
    if(resize)
        cv::resize(img, img, cv::Size(width_resize, height_resize));
    return img;
}

void View::Print(){
    std::cout<<"Frame "+std::to_string(id)+": "+name<<std::endl;
    std::cout<<"Rotation: \n"<<Rotation()<<std::endl;
    std::cout<<"Position: \n"<<Position()<<std::endl;
    std::cout<<std::endl;
}

std::vector<int> Reconstruction::ViewIds(){
    std::vector<int> retval;
    for (auto const& element : views) {
        retval.push_back(element.second.id);
    }
    return retval;
}

std::pair<int, int> Reconstruction::GetNeighboringKeyframes(int view_id){
    int previous_keyframe=-1;
    int next_keyframe=-1;
    for(int idx=view_id-1; idx>min_view_id; idx--){
        if(views.count(idx)==0){
            continue;
        }
        if(views[idx].isKeyframe()){
            previous_keyframe = idx;
            break;
        }
    }

    for(int idx=view_id+1; idx<max_view_id; idx++){
        if(views.count(idx)==0){
            continue;
        }
        if(views[idx].isKeyframe()){
            next_keyframe = idx;
            break;
        }
    }

    if(previous_keyframe<0 or next_keyframe<0){
        return std::make_pair(-1, -1);
    }
    return std::make_pair(previous_keyframe, next_keyframe);
};

std::vector<int> Reconstruction::GetReferenceFrames(int view_id){
    std::pair<int, int> kf = GetNeighboringKeyframes(view_id);
    std::vector<int> ref;
    if(kf.first==-1 and kf.second==-1)
        return ref;
    double dist = (views[kf.second].Position()- views[kf.first].Position()).norm()/2;
    Eigen::Vector3d pos = views[view_id].Position();

    for(int idx=view_id+1; idx<max_view_id; idx++){
        if(views.count(idx)==0){
            continue;
        }
        if((pos-views[idx].Position()).norm()>dist){
            ref.emplace_back(idx);
            break;
        }
    }

    for(int idx=view_id-1; idx>min_view_id; idx--){
        if(views.count(idx)==0){
            continue;
        }
        if((pos-views[idx].Position()).norm()>dist){
            ref.emplace_back(idx);
            break;
        }
    }

    return ref;
}

cv::Mat Reconstruction::GetImage(int view_id, bool resize){
    return views[view_id].GetImage(image_folder, resize);
}

Eigen::Matrix3d Reconstruction::GetQuat(int view_id){
    return views[view_id].Rotation();
}

cv::Mat Reconstruction::GetSparseDepthMap(int frame_id, bool resize){
    Camera camera = cameras[views[frame_id].camera_id];
    View view = views[frame_id];
    Eigen::Vector3d view_pos = view.Position();
    cv::Mat depth_map = cv::Mat::zeros(cv::Size(camera.width, camera.height), CV_64FC1);
    for(const auto& item : view.points2d){
        int point_id = item.first;
        std::pair<double, double> coord = item.second;
        Eigen::Vector3d pos3d = points3d[point_id].position3d;
        double depth = (pos3d-view_pos).norm();
        depth_map.at<double>(int(coord.second), int(coord.first)) = depth;
    }

    if(resize)
        cv::resize(depth_map, depth_map, cv::Size(width_resize, height_resize));

    return depth_map;
}

