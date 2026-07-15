#include "petpet_face/image_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace petpet_face
{

LetterboxResult letterbox(
    const cv::Mat &source,
    int targetWidth,
    int targetHeight)
{
    if (source.empty())
    {
        throw std::invalid_argument("Cannot letterbox an empty image");
    }
    if (targetWidth <= 0 || targetHeight <= 0)
    {
        throw std::invalid_argument("Letterbox dimensions must be positive");
    }

    const float scale = std::min(
        static_cast<float>(targetWidth) / source.cols,
        static_cast<float>(targetHeight) / source.rows);
    const int resizedWidth = static_cast<int>(std::round(source.cols * scale));
    const int resizedHeight = static_cast<int>(std::round(source.rows * scale));

    cv::Mat resized;
    cv::resize(
        source,
        resized,
        cv::Size(resizedWidth, resizedHeight),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    const int remainingWidth = targetWidth - resizedWidth;
    const int remainingHeight = targetHeight - resizedHeight;
    const int padLeft = remainingWidth / 2;
    const int padTop = remainingHeight / 2;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        padTop,
        remainingHeight - padTop,
        padLeft,
        remainingWidth - padLeft,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    return {padded, scale, padLeft, padTop};
}

} // namespace petpet_face
