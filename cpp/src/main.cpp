#include "petpet_face/face_detector.hpp"
#include "petpet_face/media_processor.hpp"
#include "petpet_face/petpet_renderer.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

std::filesystem::path findRuntimeRoot(const char *executablePath)
{
    std::vector<std::filesystem::path> candidates{
        std::filesystem::current_path(),
        std::filesystem::absolute(executablePath).parent_path()};
    for (std::filesystem::path candidate : candidates)
    {
        while (!candidate.empty())
        {
            if (std::filesystem::is_regular_file(
                    candidate / "models/face/face_yolo11s.onnx") &&
                std::filesystem::is_directory(
                    candidate / "assets/petpet_frames"))
            {
                return candidate;
            }
            const std::filesystem::path parent = candidate.parent_path();
            if (parent == candidate)
            {
                break;
            }
            candidate = parent;
        }
    }
    throw std::runtime_error(
        "Could not locate models/face and assets/petpet_frames");
}

void printUsage()
{
    std::cerr
        << "File:   petpet_face <model.onnx> <frames_dir> <input> <output> [--gpu]\n"
        << "Camera: petpet_face --camera [camera_index] [--gpu]\n";
}

} // namespace

int main(int argc, char *argv[])
{
    const bool cameraMode = argc >= 2 && std::string(argv[1]) == "--camera";
    if ((!cameraMode && (argc < 5 || argc > 6)) ||
        (cameraMode && argc > 4))
    {
        printUsage();
        return 2;
    }

    try
    {
        if (cameraMode)
        {
            int cameraIndex = 0;
            bool useCuda = false;
            for (int index = 2; index < argc; ++index)
            {
                const std::string argument(argv[index]);
                if (argument == "--gpu")
                {
                    useCuda = true;
                }
                else
                {
                    cameraIndex = std::stoi(argument);
                }
            }

            const std::filesystem::path runtimeRoot = findRuntimeRoot(argv[0]);
            petpet_face::DetectorConfig detectorConfig;
            detectorConfig.useCuda = useCuda;
            petpet_face::FaceDetector detector(
                runtimeRoot / "models/face/face_yolo11s.onnx",
                detectorConfig);
            petpet_face::PetPetRenderer renderer(
                runtimeRoot / "assets/petpet_frames");
            std::cout << "Execution provider: " << detector.executionProvider() << '\n'
                      << "Camera index: " << cameraIndex << '\n'
                      << "Press Esc or Q to exit\n";
            petpet_face::MediaProcessor processor;
            processor.processCamera(cameraIndex, detector, renderer);
            return 0;
        }

        petpet_face::DetectorConfig detectorConfig;
        detectorConfig.useCuda = argc == 6 && std::string(argv[5]) == "--gpu";

        petpet_face::FaceDetector detector(argv[1], detectorConfig);
        petpet_face::PetPetRenderer renderer(argv[2]);
        std::cout << "Execution provider: " << detector.executionProvider() << '\n'
                  << "Animation frames: " << renderer.frameCount() << '\n';

        petpet_face::MediaProcessor processor;
        processor.process(argv[3], argv[4], detector, renderer);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
