#include "petpet_face/media_processor.hpp"

#include "petpet_face/audio_muxer.hpp"
#include "petpet_face/face_detector.hpp"
#include "petpet_face/face_tracker.hpp"
#include "petpet_face/petpet_renderer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace petpet_face
{
namespace
{

std::string lowercaseExtension(const std::filesystem::path &path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

bool isImage(const std::filesystem::path &path)
{
    const std::string extension = lowercaseExtension(path);
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".bmp" || extension == ".webp";
}

struct TemporaryVideoGuard
{
    std::filesystem::path path;

    ~TemporaryVideoGuard()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

void MediaProcessor::processCamera(
    int cameraIndex,
    FaceDetector &detector,
    const PetPetRenderer &renderer) const
{
    if (cameraIndex < 0)
    {
        throw std::invalid_argument("Camera index must not be negative");
    }

    cv::VideoCapture capture;
#ifdef _WIN32
    capture.open(cameraIndex, cv::CAP_DSHOW);
    if (!capture.isOpened())
    {
        capture.release();
        capture.open(cameraIndex);
    }
#else
    capture.open(cameraIndex);
#endif
    if (!capture.isOpened())
    {
        throw std::runtime_error(
            "Failed to open camera index " + std::to_string(cameraIndex));
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280.0);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720.0);
    capture.set(cv::CAP_PROP_FPS, 30.0);
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);

    constexpr const char *windowName = "PetPet Camera - Esc/Q to exit";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    FaceTracker tracker;
    const auto startTime = std::chrono::steady_clock::now();
    auto fpsStart = startTime;
    int fpsFrameCount = 0;
    double displayFps = 0.0;

    cv::Mat frame;
    while (capture.read(frame) && !frame.empty())
    {
        cv::flip(frame, frame, 1);
        const auto faces = detector.detect(frame);
        const auto stableFaces = tracker.update(faces, frame.size());
        const auto now = std::chrono::steady_clock::now();
        const std::int64_t elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime)
                .count();
        renderer.render(frame, stableFaces, elapsedMs);

        ++fpsFrameCount;
        const double fpsElapsed =
            std::chrono::duration<double>(now - fpsStart).count();
        if (fpsElapsed >= 0.5)
        {
            displayFps = fpsFrameCount / fpsElapsed;
            fpsFrameCount = 0;
            fpsStart = now;
        }
        cv::putText(
            frame,
            "FPS " + std::to_string(static_cast<int>(std::lround(displayFps))),
            cv::Point(16, 34),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(40, 230, 40),
            2,
            cv::LINE_AA);

        cv::imshow(windowName, frame);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1.0)
        {
            break;
        }
    }
    capture.release();
    cv::destroyWindow(windowName);
}

void MediaProcessor::process(
    const std::filesystem::path &inputPath,
    const std::filesystem::path &outputPath,
    FaceDetector &detector,
    const PetPetRenderer &renderer) const
{
    if (!std::filesystem::is_regular_file(inputPath))
    {
        throw std::runtime_error("Input file does not exist: " + inputPath.string());
    }
    if (!outputPath.parent_path().empty())
    {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    if (isImage(inputPath))
    {
        cv::Mat image = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Failed to read image: " + inputPath.string());
        }
        const auto faces = detector.detect(image);
        renderer.render(image, faces, 0);
        if (!cv::imwrite(outputPath.string(), image))
        {
            throw std::runtime_error("Failed to write image: " + outputPath.string());
        }
        std::cout << "Detected faces: " << faces.size() << '\n';
        return;
    }

    cv::VideoCapture capture(inputPath.string());
    if (!capture.isOpened())
    {
        throw std::runtime_error("Failed to open input video: " + inputPath.string());
    }
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
    {
        fps = 25.0;
    }
    TemporaryVideoGuard videoOnly{makeVideoOnlyPath(outputPath)};
    cv::VideoWriter writer(
        videoOnly.path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        throw std::runtime_error("Failed to create output video: " + outputPath.string());
    }

    cv::Mat frame;
    std::int64_t frameIndex = 0;
    std::int64_t detections = 0;
    std::int64_t renderedFaces = 0;
    std::int64_t recoveredFrames = 0;
    FaceTracker tracker;
    while (capture.read(frame) && !frame.empty())
    {
        const auto faces = detector.detect(frame);
        detections += static_cast<std::int64_t>(faces.size());
        const auto stableFaces = tracker.update(faces, frame.size());
        renderedFaces += static_cast<std::int64_t>(stableFaces.size());
        if (faces.empty() && !stableFaces.empty())
        {
            ++recoveredFrames;
        }
        const std::int64_t elapsedMs = static_cast<std::int64_t>(
            std::llround(frameIndex * 1000.0 / fps));
        renderer.render(frame, stableFaces, elapsedMs);
        writer.write(frame);
        ++frameIndex;
    }
    writer.release();
    capture.release();
    mergeOriginalAudio(inputPath, videoOnly.path, outputPath);
    std::cout << "Processed frames: " << frameIndex << '\n'
              << "Raw detections: " << detections << '\n'
              << "Rendered tracked faces: " << renderedFaces << '\n'
              << "Recovered missed frames: " << recoveredFrames << '\n';
}

} // namespace petpet_face
