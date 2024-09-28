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


// 디렉토리에서 이미지를 로드하는 함수
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folderPath) {
    std::vector<cv::Mat> images;
    
    DIR* dir = opendir(folderPath.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << folderPath << std::endl;
        return images;
    }
    std::vector<std::string> fileNames;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName = entry->d_name;
        if (fileName.find(".jpg") != std::string::npos) {
            fileNames.push_back(fileName);
        }
    }
    
    // Sort file names in ascending order
    std::sort(fileNames.begin(), fileNames.end());
    
    for (const auto& fileName : fileNames) {
        std::string fullPath = folderPath + "/" + fileName;
        cv::Mat img = cv::imread(fullPath, cv::IMREAD_COLOR);
        if (!img.empty()) {
            images.push_back(img);
            // std::cout << "Loaded: " << fileName << std::endl;
        } else {
            std::cerr << "Failed to load: " << fileName << std::endl;
        }
    }
    
    closedir(dir);
    return images;
}

int main() {
    std::string extractFolderPath = "../../src/extract";
    
    if (!fileExists(extractFolderPath)) {
        std::cerr << "Error: Folder does not exist: " << extractFolderPath << std::endl;
        return 1;
    }
    
    std::vector<cv::Mat> extractedImages = loadImagesFromFolder(extractFolderPath);
    
    std::cout << "Total images loaded: " << extractedImages.size() << std::endl;
    

    cv::namedWindow("Image Viewer", cv::WINDOW_NORMAL);  // 창 생성
    for (size_t i = 0; i < extractedImages.size(); ++i) {
        if (extractedImages[i].empty()) {
            std::cerr << "Image is empty!" << std::endl;
            continue;
        }
        int originalWidth = extractedImages[i].cols;
        int originalHeight = extractedImages[i].rows;
        
        cv::Mat resizedImage;
        cv::resize(extractedImages[i], resizedImage, cv::Size(originalWidth / 2, originalHeight / 2 ));
        cv::imshow("Image Viewer", resizedImage);
            
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
