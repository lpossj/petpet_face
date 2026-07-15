#pragma once

#include <vector>

#include <opencv2/core/types.hpp>

namespace petpet_face
{

struct FaceTrackerConfig
{
    float minimumIou{0.15F};
    float smoothingFactor{0.55F};
    int maximumMissedFrames{4};
};

class FaceTracker
{
public:
    explicit FaceTracker(FaceTrackerConfig config = {});

    std::vector<cv::Rect> update(
        const std::vector<cv::Rect> &detections,
        const cv::Size &frameSize);

private:
    struct Track
    {
        cv::Rect2f box;
        cv::Vec4f velocity{};
        int missedFrames{};
    };

    static float intersectionOverUnion(
        const cv::Rect2f &first,
        const cv::Rect2f &second);

    FaceTrackerConfig config_;
    std::vector<Track> tracks_;
};

} // namespace petpet_face
