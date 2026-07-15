#include "petpet_face/audio_muxer.hpp"

#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

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

std::filesystem::path findBundledFfmpeg()
{
    std::vector<std::filesystem::path> candidates{
        std::filesystem::current_path()};
#ifdef _WIN32
    std::array<wchar_t, 32768> executablePath{};
    const DWORD length = GetModuleFileNameW(
        nullptr,
        executablePath.data(),
        static_cast<DWORD>(executablePath.size()));
    if (length > 0 && length < executablePath.size())
    {
        candidates.emplace_back(
            std::filesystem::path(executablePath.data()).parent_path());
    }
#endif

    for (std::filesystem::path candidate : candidates)
    {
        while (!candidate.empty())
        {
#ifdef _WIN32
            const auto ffmpeg = candidate / "tools/ffmpeg.exe";
#else
            const auto ffmpeg = candidate / "tools/ffmpeg";
#endif
            if (std::filesystem::is_regular_file(ffmpeg))
            {
                return ffmpeg;
            }
            const auto parent = candidate.parent_path();
            if (parent == candidate)
            {
                break;
            }
            candidate = parent;
        }
    }

#ifdef _WIN32
    std::array<wchar_t, 32768> resolved{};
    if (SearchPathW(
            nullptr,
            L"ffmpeg.exe",
            nullptr,
            static_cast<DWORD>(resolved.size()),
            resolved.data(),
            nullptr) > 0)
    {
        return resolved.data();
    }
#endif
    throw std::runtime_error(
        "FFmpeg was not found. Put ffmpeg.exe in tools/ next to the project");
}

#ifdef _WIN32
std::wstring quote(const std::filesystem::path &path)
{
    return L"\"" + path.wstring() + L"\"";
}
#endif

} // namespace

std::filesystem::path makeVideoOnlyPath(
    const std::filesystem::path &outputPath)
{
    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    return outputPath.parent_path() /
           (L"." + outputPath.stem().wstring() + L"_video_only_" +
            std::to_wstring(stamp) + L".mp4");
}

void mergeOriginalAudio(
    const std::filesystem::path &inputPath,
    const std::filesystem::path &videoOnlyPath,
    const std::filesystem::path &outputPath)
{
    const std::filesystem::path ffmpeg = findBundledFfmpeg();
#ifdef _WIN32
    std::wstring commandLine =
        quote(ffmpeg) +
        L" -y -hide_banner -loglevel error -i " + quote(videoOnlyPath) +
        L" -i " + quote(inputPath) +
        L" -map 0:v:0 -map 1:a? -c:v copy -c:a aac -b:a 192k"
        L" -movflags +faststart " + quote(outputPath);
    std::vector<wchar_t> mutableCommand(commandLine.begin(), commandLine.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(
            ffmpeg.c_str(),
            mutableCommand.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup,
            &process))
    {
        throw std::runtime_error("Failed to start FFmpeg for audio merging");
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (exitCode != 0 || !std::filesystem::is_regular_file(outputPath))
    {
        throw std::runtime_error(
            "FFmpeg failed to merge the original audio (exit code " +
            std::to_string(exitCode) + ")");
    }
#else
    (void)inputPath;
    (void)videoOnlyPath;
    (void)outputPath;
    throw std::runtime_error("Audio merging is currently supported on Windows");
#endif
}

} // namespace petpet_face
