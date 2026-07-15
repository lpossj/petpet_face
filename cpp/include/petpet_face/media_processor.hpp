#pragma once

#include <filesystem>

namespace petpet_face
{

class FaceDetector;
class PetPetRenderer;

class MediaProcessor
{
public:
    void processCamera(
        int cameraIndex,
        FaceDetector &detector,
        const PetPetRenderer &renderer) const;

    void process(
        const std::filesystem::path &inputPath,
        const std::filesystem::path &outputPath,
        FaceDetector &detector,
        const PetPetRenderer &renderer) const;
};

} // namespace petpet_face
