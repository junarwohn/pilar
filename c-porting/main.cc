#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>


// check file exists
bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}


/*
std::vector<std::string> loadImageFiles(const std::string& folder_path) {
	DIR* dir = opendir(folder_path.c_str());
	if (!dir) {
		std::cerr << "Error: Can not open directory " << folder_path << std::endl;
		return NULL;
	}

    std::vector<std::string> file_names;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string file_name = entry->d_name;
        if (file_name.find(".jpg") != std::string::npos) {
            file_names.push_back(file_name);
        }
    }

	return file_names;
}
*/
	

// 디렉토리에서 이미지를 로드하는 함수
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folder_path) {
    std::vector<cv::Mat> images;
    
    DIR* dir = opendir(folder_path.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << folder_path << std::endl;
        return images;
    }
    std::vector<std::string> file_names;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string file_name = entry->d_name;
        if (file_name.find(".jpg") != std::string::npos) {
            file_names.push_back(file_name);
        }
    }
    
    // Sort file names in ascending order
    std::sort(file_names.begin(), file_names.end());
    
    for (const auto& file_name: file_names) {
        std::string full_path = folder_path + "/" + file_name;
        cv::Mat img = cv::imread(full_path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            images.push_back(img);
            std::cout << "Loaded: " << file_name << std::endl;
        } else {
            std::cerr << "Failed to load: " << file_name << std::endl;
        }
    }
    
    closedir(dir);
    std::cout << "LOAD DONE" << std::endl;
    return images;
}

int main() {
    std::string extract_folder_path= "../../src/extract";

    std::cout << "Welcome" << std::endl;
    
    if (!fileExists(extract_folder_path)) {
        std::cerr << "Error: Folder does not exist: " << extract_folder_path << std::endl;
        return 1;
    }

    
    std::cout << "Load Images" << std::endl;
    std::vector<cv::Mat> extracted_images = loadImagesFromFolder(extract_folder_path);
    
    std::cout << "Total images loaded: " << extracted_images.size() << std::endl;
    

    cv::namedWindow("Image Viewer", cv::WINDOW_NORMAL);  // 창 생성
    for (size_t i = 0; i < extracted_images.size(); ++i) {
        if (extracted_images[i].empty()) {
            std::cerr << "Image is empty!" << std::endl;
            continue;
        }
        int original_width = extracted_images[i].cols;
        int original_height = extracted_images[i].rows;
        
        cv::Mat resized_image;
        cv::resize(extracted_images[i], resized_image, cv::Size(original_width / 2, original_height / 2 ));
        cv::imshow("Image Viewer", resized_image);
            
        int key = cv::waitKey(0);
        
        if (key == 27) { // ESC key
            break;
        } else if (key != 13) { // If not ENTER key, stay on the same image
            --i;
        }
    }
    
    cv::destroyAllWindows();
    
    return 0;
}
