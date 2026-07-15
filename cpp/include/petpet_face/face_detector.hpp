#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace petpet_face
{

struct DetectorConfig
{
    int inputWidth{640};
    int inputHeight{640};
    float confidenceThreshold{0.35F};
    float nmsThreshold{0.45F};
    int cudaDeviceId{0};
    bool useCuda{true};
};

class FaceDetector
{
public:
    FaceDetector(
        const std::filesystem::path &modelPath,
        DetectorConfig config);
    ~FaceDetector();

    FaceDetector(FaceDetector &&) noexcept;
    FaceDetector &operator=(FaceDetector &&) noexcept;
    FaceDetector(const FaceDetector &) = delete;
    FaceDetector &operator=(const FaceDetector &) = delete;

    std::vector<cv::Rect> detect(const cv::Mat &frame);
    const std::string &executionProvider() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace petpet_face
