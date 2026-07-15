#pragma once

#include <opencv2/core/mat.hpp>

namespace petpet_face
{

struct LetterboxResult
{
    cv::Mat image;
    float scale{};
    int padX{};
    int padY{};
};

LetterboxResult letterbox(
    const cv::Mat &source,
    int targetWidth,
    int targetHeight);

} // namespace petpet_face
