#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace petpet_face
{

struct PetPetConfig
{
    float widthScale{1.45F};
    float verticalOffsetRatio{0.32F};
    int frameDurationMs{60};
};

class PetPetRenderer
{
public:
    explicit PetPetRenderer(
        const std::filesystem::path &framesDirectory,
        PetPetConfig config = {});

    void render(
        cv::Mat &image,
        const std::vector<cv::Rect> &faces,
        std::int64_t elapsedMs) const;

    std::size_t frameCount() const noexcept;

private:
    static void deformFace(
        cv::Mat &image,
        const cv::Rect &face,
        std::int64_t elapsedMs,
        int frameDurationMs);

    static void alphaBlend(
        cv::Mat &background,
        const cv::Mat &foreground,
        int left,
        int top);

    std::vector<cv::Mat> frames_;
    PetPetConfig config_;
};

} // namespace petpet_face
