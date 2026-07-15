#include "petpet_face/petpet_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace petpet_face
{
namespace
{

cv::Rect makeEffectRegion(
    const cv::Rect &face,
    const cv::Size &imageSize)
{
    // 只适度扩大脸部形变范围。
    // 顶部扩展过大会把头发一起做仿射形变，产生“重复头发”。
    const int expandLeftRight =
        static_cast<int>(std::lround(face.width * 0.10));
    const int expandTop =
        static_cast<int>(std::lround(face.height * 0.05));
    const int expandBottom =
        static_cast<int>(std::lround(face.height * 0.10));

    cv::Rect expanded(
        face.x - expandLeftRight,
        face.y - expandTop,
        face.width + expandLeftRight * 2,
        face.height + expandTop + expandBottom);

    return expanded & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

} // namespace

namespace
{

void featherTransparentCanvasEdges(cv::Mat &bgra)
{
    if (bgra.empty() || bgra.channels() != 4)
    {
        return;
    }

    // 按手部画布尺寸计算羽化宽度。只修改 Alpha，不模糊手本身。
    const int minimumSide = std::min(bgra.cols, bgra.rows);
    const int featherWidth = std::clamp(
        static_cast<int>(std::lround(minimumSide * 0.055)),
        6,
        28);

    for (int y = 0; y < bgra.rows; ++y)
    {
        cv::Vec4b *row = bgra.ptr<cv::Vec4b>(y);
        const int distanceTop = y;
        const int distanceBottom = bgra.rows - 1 - y;

        for (int x = 0; x < bgra.cols; ++x)
        {
            const int distanceLeft = x;
            const int distanceRight = bgra.cols - 1 - x;
            const int edgeDistance = std::min({
                distanceLeft,
                distanceRight,
                distanceTop,
                distanceBottom});

            if (edgeDistance >= featherWidth)
            {
                continue;
            }

            float progress = static_cast<float>(edgeDistance) /
                             static_cast<float>(featherWidth);

            // smoothstep，避免线性羽化产生可见亮边。
            progress = progress * progress * (3.0F - 2.0F * progress);

            row[x][3] = cv::saturate_cast<unsigned char>(
                static_cast<float>(row[x][3]) * progress);
        }
    }
}

} // namespace

PetPetRenderer::PetPetRenderer(
    const std::filesystem::path &framesDirectory,
    PetPetConfig config)
    : config_(config)
{
    if (!std::filesystem::is_directory(framesDirectory))
    {
        throw std::runtime_error(
            "PetPet frames directory does not exist: " + framesDirectory.string());
    }
    if (config_.widthScale <= 0.0F || config_.frameDurationMs <= 0)
    {
        throw std::invalid_argument("Invalid PetPet renderer configuration");
    }

    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(framesDirectory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    for (const auto &path : paths)
    {
        cv::Mat frame = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (frame.empty())
        {
            throw std::runtime_error("Failed to read PetPet frame: " + path.string());
        }
        if (frame.channels() != 4)
        {
            throw std::runtime_error(
                "PetPet frame must be transparent BGRA PNG: " + path.string());
        }
        frames_.push_back(std::move(frame));
    }
    if (frames_.empty())
    {
        throw std::runtime_error(
            "No PNG animation frames found in: " + framesDirectory.string());
    }
}

void PetPetRenderer::render(
    cv::Mat &image,
    const std::vector<cv::Rect> &faces,
    std::int64_t elapsedMs) const
{
    if (image.empty() || faces.empty())
    {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(
        std::max<std::int64_t>(0, elapsedMs) / config_.frameDurationMs) % frames_.size();
    const cv::Mat &animationFrame = frames_[index];

    for (const cv::Rect &face : faces)
    {
        const cv::Rect effectRegion =
            makeEffectRegion(face, image.size());

        // 形变只适度扩大到脸部边缘，避免头发被重复形变。
        deformFace(
            image,
            effectRegion,
            elapsedMs,
            config_.frameDurationMs);

        // 原手部素材相对检测框偏大，这里缩小约 18%。
        constexpr float handSizeAdjustment = 0.82F;
        const int width = std::max(
            1,
            static_cast<int>(std::lround(
                face.width *
                config_.widthScale *
                handSizeAdjustment)));
        const int height = std::max(
            1,
            static_cast<int>(std::lround(
                width *
                static_cast<double>(animationFrame.rows) /
                animationFrame.cols)));

        cv::Mat resized;
        cv::resize(
            animationFrame,
            resized,
            cv::Size(width, height),
            0.0,
            0.0,
            cv::INTER_AREA);

        // 某些 PetPet PNG 的画布最外圈并非完全透明。
        // 对缩放后的 Alpha 边缘做羽化，隐藏矩形画布边界。
        featherTransparentCanvasEdges(resized);

        // 手仍以原始人脸中心对齐，避免扩展框导致左右漂移。
        const int left =
            face.x + face.width / 2 - width / 2;

        // 在原配置基础上额外上移，让手落在头顶而不是眼睛和脸颊。
        constexpr float extraUpwardOffset = 0.08F;
        const int top =
            effectRegion.y -
            static_cast<int>(std::lround(
                height *
                (config_.verticalOffsetRatio +
                 extraUpwardOffset)));

        alphaBlend(image, resized, left, top);
    }
}

void PetPetRenderer::deformFace(
    cv::Mat &image,
    const cv::Rect &face,
    std::int64_t elapsedMs,
    int frameDurationMs)
{
    struct Transform
    {
        float scaleX;
        float scaleY;
    };
    static constexpr std::array<Transform, 5> transforms{{
        {1.00F, 1.00F},
        {1.04F, 0.88F},
        {1.10F, 0.72F},
        {1.05F, 0.86F},
        {1.00F, 1.00F},
    }};

    const cv::Rect bounds(0, 0, image.cols, image.rows);
    const cv::Rect region = face & bounds;
    if (region.width < 4 || region.height < 4)
    {
        return;
    }

    const double safeTime = static_cast<double>(std::max<std::int64_t>(0, elapsedMs));
    const double framePosition = safeTime / frameDurationMs;
    const std::size_t currentIndex =
        static_cast<std::size_t>(std::floor(framePosition)) % transforms.size();
    const std::size_t nextIndex = (currentIndex + 1) % transforms.size();
    const float linearProgress = static_cast<float>(
        framePosition - std::floor(framePosition));
    const float progress = linearProgress * linearProgress *
                           (3.0F - 2.0F * linearProgress);
    const Transform &current = transforms[currentIndex];
    const Transform &next = transforms[nextIndex];
    const Transform transform{
        current.scaleX + (next.scaleX - current.scaleX) * progress,
        current.scaleY + (next.scaleY - current.scaleY) * progress};
    if (transform.scaleX == 1.0F && transform.scaleY == 1.0F)
    {
        return;
    }

    const cv::Mat source = image(region).clone();
    cv::Mat deformed;
    const double translateX = region.width * (1.0 - transform.scaleX) * 0.5;
    // Anchor the bottom edge. Vertical compression therefore pushes the
    // forehead downward instead of pulling the chin upward.
    const double translateY = region.height * (1.0 - transform.scaleY);
    const cv::Mat affine = (cv::Mat_<double>(2, 3) <<
        transform.scaleX, 0.0, translateX,
        0.0, transform.scaleY, translateY);
    cv::warpAffine(
        source,
        deformed,
        affine,
        source.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    cv::Mat mask(source.size(), CV_8UC1, cv::Scalar(0));
    const cv::Point center(region.width / 2, region.height / 2);
    const cv::Size axes(
        std::max(1, static_cast<int>(region.width * 0.46)),
        std::max(1, static_cast<int>(region.height * 0.48)));
    cv::ellipse(mask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);

    // Scale the feather with the detected face. A fixed 7x7 blur is almost
    // invisible on a large camera frame and leaves a sharp moving boundary.
    const int minimumSide = std::min(region.width, region.height);
    int featherKernel = std::max(
        9,
        static_cast<int>(std::lround(minimumSide * 0.18)));
    if (featherKernel % 2 == 0)
    {
        ++featherKernel;
    }
    int largestValidKernel = minimumSide;
    if (largestValidKernel % 2 == 0)
    {
        --largestValidKernel;
    }
    featherKernel = std::max(1, std::min(featherKernel, largestValidKernel));
    cv::GaussianBlur(
        mask,
        mask,
        cv::Size(featherKernel, featherKernel),
        featherKernel * 0.24,
        featherKernel * 0.24,
        cv::BORDER_REFLECT_101);

    cv::Mat destination = image(region);
    for (int y = 0; y < destination.rows; ++y)
    {
        const cv::Vec3b *deformedRow = deformed.ptr<cv::Vec3b>(y);
        cv::Vec3b *destinationRow = destination.ptr<cv::Vec3b>(y);
        const unsigned char *maskRow = mask.ptr<unsigned char>(y);
        for (int x = 0; x < destination.cols; ++x)
        {
            const float alpha = maskRow[x] / 255.0F;
            for (int channel = 0; channel < 3; ++channel)
            {
                destinationRow[x][channel] = cv::saturate_cast<unsigned char>(
                    deformedRow[x][channel] * alpha +
                    destinationRow[x][channel] * (1.0F - alpha));
            }
        }
    }
}

std::size_t PetPetRenderer::frameCount() const noexcept
{
    return frames_.size();
}

void PetPetRenderer::alphaBlend(
    cv::Mat &background,
    const cv::Mat &foreground,
    int left,
    int top)
{
    const cv::Rect canvas(0, 0, background.cols, background.rows);
    const cv::Rect requested(left, top, foreground.cols, foreground.rows);
    const cv::Rect destination = requested & canvas;
    if (destination.empty())
    {
        return;
    }

    const cv::Rect source(
        destination.x - left,
        destination.y - top,
        destination.width,
        destination.height);
    cv::Mat backgroundRoi = background(destination);
    cv::Mat foregroundRoi = foreground(source);

    for (int y = 0; y < destination.height; ++y)
    {
        const cv::Vec4b *foregroundRow = foregroundRoi.ptr<cv::Vec4b>(y);
        cv::Vec3b *backgroundRow = backgroundRoi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < destination.width; ++x)
        {
            const float alpha = foregroundRow[x][3] / 255.0F;
            for (int channel = 0; channel < 3; ++channel)
            {
                backgroundRow[x][channel] = cv::saturate_cast<unsigned char>(
                    foregroundRow[x][channel] * alpha +
                    backgroundRow[x][channel] * (1.0F - alpha));
            }
        }
    }
}

} // namespace petpet_face
