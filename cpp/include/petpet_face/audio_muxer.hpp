#pragma once

#include <filesystem>

namespace petpet_face
{

std::filesystem::path makeVideoOnlyPath(
    const std::filesystem::path &outputPath);

void mergeOriginalAudio(
    const std::filesystem::path &inputPath,
    const std::filesystem::path &videoOnlyPath,
    const std::filesystem::path &outputPath);

} // namespace petpet_face
