#include "petpet_face/face_detector.hpp"

#include "petpet_face/image_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace petpet_face
{
namespace
{

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    return options;
}

void configureCudaRuntimeSearchPath()
{
#if defined(_WIN32)
    std::array<wchar_t, 32768> executablePath{};
    const DWORD length = GetModuleFileNameW(
        nullptr,
        executablePath.data(),
        static_cast<DWORD>(executablePath.size()));
    if (length == 0 || length >= executablePath.size())
    {
        return;
    }

    const std::filesystem::path runtimeDirectory =
        std::filesystem::path(executablePath.data()).parent_path() / "gpu";
    if (std::filesystem::is_directory(runtimeDirectory) &&
        !SetDllDirectoryW(runtimeDirectory.c_str()))
    {
        std::cerr
            << "Warning: failed to register CUDA runtime directory: "
            << runtimeDirectory.string() << '\n';
    }
#endif
}

bool providerIsAvailable(const std::string &providerName)
{
    const auto providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(), providerName) != providers.end();
}

} // namespace

struct FaceDetector::Impl
{
    explicit Impl(
        const std::filesystem::path &modelPath,
        DetectorConfig detectorConfig)
        : config(std::move(detectorConfig)),
          environment(ORT_LOGGING_LEVEL_WARNING, "petpet_face"),
          session(nullptr)
    {
        configureCudaRuntimeSearchPath();

        if (!std::filesystem::is_regular_file(modelPath))
        {
            throw std::runtime_error("Model file does not exist: " + modelPath.string());
        }
        if (config.inputWidth <= 0 || config.inputHeight <= 0)
        {
            throw std::invalid_argument("Model input dimensions must be positive");
        }

        if (config.useCuda)
        {
            if (!providerIsAvailable("CUDAExecutionProvider"))
            {
                throw std::runtime_error(
                    "CUDAExecutionProvider is unavailable in this ONNX Runtime build");
            }
            try
            {
                Ort::SessionOptions options = makeSessionOptions();
                OrtCUDAProviderOptions cudaOptions{};
                cudaOptions.device_id = config.cudaDeviceId;
                cudaOptions.arena_extend_strategy = 0;
                cudaOptions.gpu_mem_limit = std::numeric_limits<std::size_t>::max();
                cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cudaOptions.do_copy_in_default_stream = 1;
                options.AppendExecutionProvider_CUDA(cudaOptions);
                session = Ort::Session(environment, modelPath.c_str(), options);
                provider = "CUDAExecutionProvider";
            }
            catch (const Ort::Exception &error)
            {
                throw std::runtime_error(
                    std::string("CUDA initialization failed: ") + error.what());
            }
        }
        else
        {
            Ort::SessionOptions options = makeSessionOptions();
            session = Ort::Session(environment, modelPath.c_str(), options);
            provider = "CPUExecutionProvider";
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const Ort::AllocatedStringPtr allocatedInputName =
            session.GetInputNameAllocated(0, allocator);
        const Ort::AllocatedStringPtr allocatedOutputName =
            session.GetOutputNameAllocated(0, allocator);
        inputName = allocatedInputName.get();
        outputName = allocatedOutputName.get();
    }

    std::vector<cv::Rect> detect(const cv::Mat &frame)
    {
        if (frame.empty())
        {
            return {};
        }

        const LetterboxResult prepared =
            letterbox(frame, config.inputWidth, config.inputHeight);
        cv::Mat blob = cv::dnn::blobFromImage(
            prepared.image,
            1.0 / 255.0,
            cv::Size(config.inputWidth, config.inputHeight),
            cv::Scalar(),
            true,
            false,
            CV_32F);
        if (!blob.isContinuous())
        {
            blob = blob.clone();
        }

        const std::array<int64_t, 4> inputShape{
            1, 3, config.inputHeight, config.inputWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            blob.ptr<float>(),
            blob.total(),
            inputShape.data(),
            inputShape.size());

        const char *inputNames[]{inputName.c_str()};
        const char *outputNames[]{outputName.c_str()};
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        if (outputs.empty() || !outputs.front().IsTensor())
        {
            throw std::runtime_error("The model returned no tensor output");
        }

        const std::vector<int64_t> outputShape =
            outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() != 3 || outputShape[0] != 1 || outputShape[1] != 5)
        {
            std::string dimensions;
            for (const int64_t dimension : outputShape)
            {
                dimensions += " " + std::to_string(dimension);
            }
            throw std::runtime_error("Unexpected model output shape:" + dimensions);
        }

        const int64_t candidateCount = outputShape[2];
        const float *data = outputs.front().GetTensorData<float>();
        std::vector<cv::Rect> candidateBoxes;
        std::vector<float> candidateScores;
        candidateBoxes.reserve(static_cast<std::size_t>(candidateCount));
        candidateScores.reserve(static_cast<std::size_t>(candidateCount));

        for (int64_t index = 0; index < candidateCount; ++index)
        {
            const float confidence = data[4 * candidateCount + index];
            if (confidence < config.confidenceThreshold)
            {
                continue;
            }

            const float centerX = data[index];
            const float centerY = data[candidateCount + index];
            const float boxWidth = data[2 * candidateCount + index];
            const float boxHeight = data[3 * candidateCount + index];
            const int left = static_cast<int>(std::floor(
                (centerX - boxWidth / 2.0F - prepared.padX) / prepared.scale));
            const int top = static_cast<int>(std::floor(
                (centerY - boxHeight / 2.0F - prepared.padY) / prepared.scale));
            const int right = static_cast<int>(std::ceil(
                (centerX + boxWidth / 2.0F - prepared.padX) / prepared.scale));
            const int bottom = static_cast<int>(std::ceil(
                (centerY + boxHeight / 2.0F - prepared.padY) / prepared.scale));
            cv::Rect box(left, top, right - left, bottom - top);
            box &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (box.width > 1 && box.height > 1)
            {
                candidateBoxes.push_back(box);
                candidateScores.push_back(confidence);
            }
        }

        std::vector<int> selectedIndices;
        cv::dnn::NMSBoxes(
            candidateBoxes,
            candidateScores,
            config.confidenceThreshold,
            config.nmsThreshold,
            selectedIndices);
        std::vector<cv::Rect> finalBoxes;
        finalBoxes.reserve(selectedIndices.size());
        for (const int index : selectedIndices)
        {
            finalBoxes.push_back(candidateBoxes[static_cast<std::size_t>(index)]);
        }
        return finalBoxes;
    }

    DetectorConfig config;
    Ort::Env environment;
    Ort::Session session;
    std::string inputName;
    std::string outputName;
    std::string provider;
};

FaceDetector::FaceDetector(
    const std::filesystem::path &modelPath,
    DetectorConfig config)
    : impl_(std::make_unique<Impl>(modelPath, std::move(config)))
{
}

FaceDetector::~FaceDetector() = default;
FaceDetector::FaceDetector(FaceDetector &&) noexcept = default;
FaceDetector &FaceDetector::operator=(FaceDetector &&) noexcept = default;

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat &frame)
{
    return impl_->detect(frame);
}

const std::string &FaceDetector::executionProvider() const noexcept
{
    return impl_->provider;
}

} // namespace petpet_face
