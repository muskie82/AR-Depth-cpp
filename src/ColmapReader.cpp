#include "ColmapReader.h"


size_t ColmapReader::split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs.size();
}

std::map<int, Camera> ColmapReader::ReadColmapCamera(std::string filename) {
    std::map<int, Camera> cameras;
    std::ifstream ifs(filename);
    std::string str;
    if (ifs.fail()) {
        std::cerr << "Failed to read camera file" << std::endl;
        return cameras;
    }
    while (getline(ifs, str)) {
        std::cout << "[" << str << "]" << std::endl;
        std::vector <std::string> tokens;
        split(str, tokens, ' ');
        int id_value = std::stoi(tokens[0]);
        cameras[id_value] = Camera();
        cameras[id_value].id = id_value;
        cameras[id_value].model = tokens[1];
        assert(cameras[id_value].model == "PINHOLE");
        cameras[id_value].width = std::stoi(tokens[2]);
        cameras[id_value].height = std::stoi(tokens[3]);
        cameras[id_value].fx = std::stod(tokens[4]);
        cameras[id_value].fy = std::stod(tokens[5]);
        cameras[id_value].cx = std::stod(tokens[6]);
        cameras[id_value].cy = std::stod(tokens[7]);
    }
    return cameras;

};

std::map<int, View> ColmapReader::ReadColmapImages(std::string filename) {
    std::map<int, Point> points;
    std::ifstream ifs(filename);
    std::string str;
    std::map<int, View> views;
    while (getline(ifs, str)) {
        std::vector <std::string> tokens;
        split(str, tokens, ' ');
        int id_value = std::stoi(tokens[0]);
        views[id_value] = View();
        views[id_value].id = id_value;
        views[id_value].orientation = Eigen::Quaterniond(std::stod(tokens[1]), std::stod(tokens[2]),
                                                         std::stod(tokens[3]), std::stod(tokens[4]));
        views[id_value].translation = Eigen::Vector3d(std::stod(tokens[5]), std::stod(tokens[6]),
                                                      std::stod(tokens[7]));
        views[id_value].camera_id = std::stoi(tokens[8]);
        views[id_value].name = tokens[9];

        getline(ifs, str);
        split(str, tokens, ' ');

        for (int idx = 0; idx < tokens.size() / 3; idx++) {
            int point_id = std::stoi(tokens[idx * 3 + 2]);
            views[id_value].points2d[point_id] = std::make_pair(std::stod(tokens[idx * 3 + 0]),
                                                                std::stod(tokens[idx * 3 + 1]));
        }
    };

    return views;
}

std::map<int, Point> ColmapReader::ReadColmapPoints(std::string filename) {
    std::map<int, Point> points;
    std::ifstream ifs(filename);
    std::string str;
    if (ifs.fail()) {
        std::cerr << "Failed to read point file" << std::endl;
        return points;
    }

    while (getline(ifs, str)) {
        std::vector <std::string> tokens;
        split(str, tokens, ' ');
        int id_value = std::stoi(tokens[0]);
        points[id_value] = Point();
        points[id_value].id = id_value;
        points[id_value].position3d = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]),
                                                      std::stod(tokens[3]));
    }

    return points;

};

Reconstruction ColmapReader::ReadColmap(const std::string &poses_folder, const std::string &images_folder) {
    Reconstruction recon;
    recon.image_folder = images_folder;
    recon.cameras = ReadColmapCamera(poses_folder + "/cameras.txt");
    recon.views = ReadColmapImages(poses_folder + "/images.txt");
    recon.points3d = ReadColmapPoints(poses_folder + "/points3D.txt");
    recon.min_view_id = (recon.views.begin()->first);
    recon.max_view_id = ((--recon.views.end())->first);
    std::cout << "Number of points: " << recon.points3d.size() << std::endl;
    std::cout << "Number of frames: " << recon.views.size() << std::endl;
    return recon;

}