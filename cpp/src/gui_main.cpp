#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <dshow.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <winhttp.h>

#include "petpet_face/audio_muxer.hpp"
#include "petpet_face/face_detector.hpp"
#include "petpet_face/face_tracker.hpp"
#include "petpet_face/petpet_renderer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

    constexpr int ID_INPUT_EDIT = 101;
    constexpr int ID_INPUT_BROWSE = 102;
    constexpr int ID_OUTPUT_EDIT = 103;
    constexpr int ID_OUTPUT_BROWSE = 104;
    constexpr int ID_PROCESS = 105;
    constexpr int ID_CAMERA = 106;
    constexpr int ID_EFFECT = 107;
    constexpr int ID_GPU = 108;
    constexpr int ID_TARGET_FPS = 109;
    constexpr int ID_PROGRESS = 110;
    constexpr int ID_SETTINGS = 111;
    constexpr int ID_SETTINGS_CAMERA = 201;
    constexpr int ID_SETTINGS_MICROPHONE = 202;
    constexpr int ID_SETTINGS_REFRESH = 203;
    constexpr int ID_SETTINGS_SAVE = 204;
    constexpr int ID_SETTINGS_CANCEL = 205;
    constexpr int ID_SETTINGS_CAMERA_HOTKEY = 206;
    constexpr int ID_SETTINGS_RECORDING_HOTKEY = 207;
    constexpr int ID_SETTINGS_GPU_DOWNLOAD = 208;
    constexpr int ID_SETTINGS_GPU_PROGRESS = 209;

    constexpr const wchar_t *GPU_COMPONENTS_URL =
        L"https://github.com/lpossj/optionalGPUcomponents/releases/download/v1.0.0/GPU-Runtime-CUDA12-cuDNN9-x64.zip";

    constexpr UINT WM_APP_PROGRESS = WM_APP + 1;
    constexpr UINT WM_APP_COMPLETE = WM_APP + 2;
    constexpr UINT WM_APP_CAMERA_STATE = WM_APP + 3;
    constexpr UINT WM_APP_RECORDING_STATE = WM_APP + 4;
    constexpr UINT WM_APP_GPU_INSTALL_COMPLETE = WM_APP + 5;
    constexpr UINT WM_APP_GPU_INSTALL_PROGRESS = WM_APP + 6;
    constexpr UINT TIMER_REFRESH = 1;

    constexpr RECT DEFAULT_PREVIEW_RECT{366, 82, 1090, 650};

    constexpr COLORREF THEME_BACKGROUND = RGB(14, 17, 23);
    constexpr COLORREF THEME_CARD = RGB(24, 29, 39);
    constexpr COLORREF THEME_CONTROL = RGB(31, 37, 49);
    constexpr COLORREF THEME_BORDER = RGB(53, 61, 78);
    constexpr COLORREF THEME_TEXT = RGB(232, 235, 242);
    constexpr COLORREF THEME_MUTED = RGB(151, 160, 178);
    constexpr COLORREF THEME_ACCENT = RGB(124, 92, 252);
    constexpr COLORREF THEME_CAMERA = RGB(35, 190, 165);

    struct ProgressUpdate
    {
        int completed{};
        int total{};
        double fps{};
        double remainingSeconds{};
    };

    struct CompletionUpdate
    {
        bool success{};
        std::wstring message;
    };

    struct CameraStateUpdate
    {
        bool running{};
        std::wstring message;
    };

    struct RecordingStateUpdate
    {
        bool active{};
        bool success{};
        std::wstring message;
    };

    struct HotkeyBinding
    {
        UINT virtualKey{};
        BYTE modifiers{};
    };

    struct AppState
    {
        HWND window{};
        HWND inputEdit{};
        HWND outputEdit{};
        HWND processButton{};
        HWND cameraButton{};
        HWND effectCheck{};
        HWND gpuCheck{};
        HWND fpsCombo{};
        HWND progressBar{};
        HWND progressLabel{};
        HWND remainingLabel{};
        HWND actualFpsLabel{};
        HWND statusLabel{};
        HFONT font{};
        HFONT titleFont{};
        HFONT smallFont{};
        HBRUSH backgroundBrush{};
        HBRUSH cardBrush{};
        HBRUSH controlBrush{};

        RECT previewRect{DEFAULT_PREVIEW_RECT};
        int sidebarBottom{658};

        std::filesystem::path runtimeRoot;
        std::filesystem::path inputPath;
        int selectedCameraIndex{};
        std::wstring selectedCameraName;
        std::wstring selectedMicrophoneName;
        HotkeyBinding cameraHotkey{'Q', 0};
        HotkeyBinding recordingHotkey{'R', 0};
        std::atomic_bool cameraRunning{false};
        std::atomic_bool recordingRequested{false};
        std::atomic_bool recordingActive{false};
        std::atomic_bool processing{false};
        std::atomic_bool cancelProcessing{false};
        std::atomic_bool effectEnabled{true};
        std::atomic<double> actualFps{0.0};
        int targetFps{30};

        std::thread cameraThread;
        std::thread processingThread;
        std::mutex previewMutex;
        cv::Mat previewFrame;
    };

    std::filesystem::path findRuntimeRoot()
    {
        std::vector<std::filesystem::path> candidates{std::filesystem::current_path()};
        std::vector<wchar_t> executablePath(32768);
        const DWORD length = GetModuleFileNameW(
            nullptr, executablePath.data(), static_cast<DWORD>(executablePath.size()));
        if (length > 0 && length < executablePath.size())
        {
            candidates.push_back(
                std::filesystem::path(executablePath.data()).parent_path());
        }

        for (std::filesystem::path candidate : candidates)
        {
            while (!candidate.empty())
            {
                if (std::filesystem::is_regular_file(
                        candidate / "models/face/face_yolo11s.onnx") &&
                    std::filesystem::is_directory(candidate / "assets/petpet_frames"))
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
        throw std::runtime_error("Runtime model and animation assets were not found");
    }

    std::vector<std::wstring> enumerateDeviceNames(const CLSID &category)
    {
        std::vector<std::wstring> names;
        ICreateDevEnum *deviceEnumerator = nullptr;
        IEnumMoniker *monikerEnumerator = nullptr;
        if (FAILED(CoCreateInstance(
                CLSID_SystemDeviceEnum,
                nullptr,
                CLSCTX_INPROC_SERVER,
                IID_ICreateDevEnum,
                reinterpret_cast<void **>(&deviceEnumerator))) ||
            deviceEnumerator == nullptr)
        {
            return names;
        }

        if (deviceEnumerator->CreateClassEnumerator(
                category,
                &monikerEnumerator,
                0) == S_OK &&
            monikerEnumerator != nullptr)
        {
            IMoniker *moniker = nullptr;
            ULONG fetched = 0;
            while (monikerEnumerator->Next(1, &moniker, &fetched) == S_OK)
            {
                IPropertyBag *propertyBag = nullptr;
                if (SUCCEEDED(moniker->BindToStorage(
                        nullptr,
                        nullptr,
                        IID_IPropertyBag,
                        reinterpret_cast<void **>(&propertyBag))) &&
                    propertyBag != nullptr)
                {
                    VARIANT friendlyName;
                    VariantInit(&friendlyName);
                    if (SUCCEEDED(propertyBag->Read(
                            L"FriendlyName", &friendlyName, nullptr)) &&
                        friendlyName.vt == VT_BSTR &&
                        friendlyName.bstrVal != nullptr)
                    {
                        names.emplace_back(friendlyName.bstrVal);
                    }
                    VariantClear(&friendlyName);
                    propertyBag->Release();
                }
                moniker->Release();
                moniker = nullptr;
            }
            monikerEnumerator->Release();
        }
        deviceEnumerator->Release();
        return names;
    }

    constexpr wchar_t DEFAULT_MICROPHONE_SETTING[] = L"__SYSTEM_DEFAULT__";
    constexpr wchar_t NO_MICROPHONE_SETTING[] = L"__NONE__";

    std::wstring quoteCommandArgument(const std::wstring &argument)
    {
        std::wstring quoted{L"\""};
        std::size_t backslashes = 0;
        for (const wchar_t character : argument)
        {
            if (character == L'\\')
            {
                ++backslashes;
            }
            else if (character == L'\"')
            {
                quoted.append(backslashes * 2 + 1, L'\\');
                quoted.push_back(L'\"');
                backslashes = 0;
            }
            else
            {
                quoted.append(backslashes, L'\\');
                backslashes = 0;
                quoted.push_back(character);
            }
        }
        quoted.append(backslashes * 2, L'\\');
        quoted.push_back(L'\"');
        return quoted;
    }

    std::wstring quotePowerShellLiteral(const std::wstring &value)
    {
        std::wstring quoted{L"'"};
        for (const wchar_t character : value)
        {
            quoted.push_back(character);
            if (character == L'\'')
            {
                quoted.push_back(L'\'');
            }
        }
        quoted.push_back(L'\'');
        return quoted;
    }

    enum class GpuInstallResult : WPARAM
    {
        failed = 0,
        succeeded = 1,
        cancelled = 2
    };

    std::filesystem::path makeGpuTemporaryPath(const wchar_t *suffix)
    {
        std::array<wchar_t, 32768> temporaryDirectory{};
        const DWORD length = GetTempPathW(
            static_cast<DWORD>(temporaryDirectory.size()),
            temporaryDirectory.data());
        const std::filesystem::path root =
            length > 0 && length < temporaryDirectory.size()
                ? std::filesystem::path(temporaryDirectory.data())
                : std::filesystem::temp_directory_path();
        return root /
               (L"PetPetFace-GPU-" + std::to_wstring(GetCurrentProcessId()) +
                L"-" + std::to_wstring(GetTickCount64()) + suffix);
    }

    GpuInstallResult downloadGpuPackage(
        const std::filesystem::path &zipPath,
        std::atomic_bool &cancelRequested,
        HWND notifyWindow)
    {
        const std::wstring url(GPU_COMPONENTS_URL);
        URL_COMPONENTSW components{};
        components.dwStructSize = sizeof(components);
        components.dwHostNameLength = static_cast<DWORD>(-1);
        components.dwUrlPathLength = static_cast<DWORD>(-1);
        components.dwExtraInfoLength = static_cast<DWORD>(-1);
        if (!WinHttpCrackUrl(url.c_str(), 0, 0, &components) ||
            components.nScheme != INTERNET_SCHEME_HTTPS)
        {
            return GpuInstallResult::failed;
        }
        const std::wstring host(
            components.lpszHostName, components.dwHostNameLength);
        std::wstring resource(
            components.lpszUrlPath, components.dwUrlPathLength);
        if (components.dwExtraInfoLength > 0)
        {
            resource.append(
                components.lpszExtraInfo, components.dwExtraInfoLength);
        }

        HINTERNET session = WinHttpOpen(
            L"PetPetFace/0.0.2",
            WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
            WINHTTP_NO_PROXY_NAME,
            WINHTTP_NO_PROXY_BYPASS,
            0);
        if (session == nullptr)
        {
            return GpuInstallResult::failed;
        }
        WinHttpSetTimeouts(session, 5000, 5000, 5000, 2000);
        HINTERNET connection = WinHttpConnect(
            session, host.c_str(), components.nPort, 0);
        HINTERNET request = connection == nullptr
                                ? nullptr
                                : WinHttpOpenRequest(
                                      connection,
                                      L"GET",
                                      resource.c_str(),
                                      nullptr,
                                      WINHTTP_NO_REFERER,
                                      WINHTTP_DEFAULT_ACCEPT_TYPES,
                                      WINHTTP_FLAG_SECURE);
        const auto closeInternetHandles = [&]() {
            if (request != nullptr)
            {
                WinHttpCloseHandle(request);
            }
            if (connection != nullptr)
            {
                WinHttpCloseHandle(connection);
            }
            WinHttpCloseHandle(session);
        };
        if (request == nullptr ||
            !WinHttpSendRequest(
                request,
                WINHTTP_NO_ADDITIONAL_HEADERS,
                0,
                WINHTTP_NO_REQUEST_DATA,
                0,
                0,
                0) ||
            !WinHttpReceiveResponse(request, nullptr))
        {
            closeInternetHandles();
            return cancelRequested.load()
                       ? GpuInstallResult::cancelled
                       : GpuInstallResult::failed;
        }

        DWORD statusCode = 0;
        DWORD statusSize = sizeof(statusCode);
        if (!WinHttpQueryHeaders(
                request,
                WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                WINHTTP_HEADER_NAME_BY_INDEX,
                &statusCode,
                &statusSize,
                WINHTTP_NO_HEADER_INDEX) ||
            statusCode < 200 || statusCode >= 300)
        {
            closeInternetHandles();
            return GpuInstallResult::failed;
        }

        ULONGLONG totalBytes = 0;
        std::array<wchar_t, 64> contentLength{};
        DWORD contentLengthSize =
            static_cast<DWORD>(contentLength.size() * sizeof(wchar_t));
        if (WinHttpQueryHeaders(
                request,
                WINHTTP_QUERY_CONTENT_LENGTH,
                WINHTTP_HEADER_NAME_BY_INDEX,
                contentLength.data(),
                &contentLengthSize,
                WINHTTP_NO_HEADER_INDEX))
        {
            totalBytes = _wcstoui64(contentLength.data(), nullptr, 10);
        }

        HANDLE output = CreateFileW(
            zipPath.c_str(),
            GENERIC_WRITE,
            0,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_TEMPORARY,
            nullptr);
        if (output == INVALID_HANDLE_VALUE)
        {
            closeInternetHandles();
            return GpuInstallResult::failed;
        }

        std::vector<std::uint8_t> buffer(1024 * 1024);
        ULONGLONG downloadedBytes = 0;
        int lastProgress = -1;
        GpuInstallResult result = GpuInstallResult::succeeded;
        while (!cancelRequested.load())
        {
            DWORD bytesRead = 0;
            if (!WinHttpReadData(
                    request,
                    buffer.data(),
                    static_cast<DWORD>(buffer.size()),
                    &bytesRead))
            {
                result = cancelRequested.load()
                             ? GpuInstallResult::cancelled
                             : GpuInstallResult::failed;
                break;
            }
            if (bytesRead == 0)
            {
                break;
            }
            DWORD bytesWritten = 0;
            if (!WriteFile(
                    output,
                    buffer.data(),
                    bytesRead,
                    &bytesWritten,
                    nullptr) ||
                bytesWritten != bytesRead)
            {
                result = GpuInstallResult::failed;
                break;
            }
            downloadedBytes += bytesRead;
            const int progress =
                totalBytes > 0
                    ? static_cast<int>(std::min<ULONGLONG>(
                          90, downloadedBytes * 90 / totalBytes))
                    : 0;
            if (progress != lastProgress)
            {
                lastProgress = progress;
                PostMessageW(
                    notifyWindow,
                    WM_APP_GPU_INSTALL_PROGRESS,
                    static_cast<WPARAM>(progress),
                    0);
            }
        }
        if (cancelRequested.load())
        {
            result = GpuInstallResult::cancelled;
        }
        CloseHandle(output);
        closeInternetHandles();
        if (result != GpuInstallResult::succeeded)
        {
            std::error_code removeError;
            std::filesystem::remove(zipPath, removeError);
        }
        return result;
    }

    GpuInstallResult installGpuComponents(
        const std::filesystem::path &runtimeRoot,
        std::atomic_bool &cancelRequested,
        HWND notifyWindow)
    {
        const std::filesystem::path zipPath =
            makeGpuTemporaryPath(L".zip");
        const std::filesystem::path stagePath =
            makeGpuTemporaryPath(L"-stage");
        const GpuInstallResult downloadResult = downloadGpuPackage(
            zipPath, cancelRequested, notifyWindow);
        if (downloadResult != GpuInstallResult::succeeded)
        {
            return downloadResult;
        }
        PostMessageW(notifyWindow, WM_APP_GPU_INSTALL_PROGRESS, 92, 0);

        std::array<wchar_t, MAX_PATH> systemDirectory{};
        const UINT systemLength = GetSystemDirectoryW(
            systemDirectory.data(), static_cast<UINT>(systemDirectory.size()));
        if (systemLength == 0 || systemLength >= systemDirectory.size())
        {
            return GpuInstallResult::failed;
        }
        const std::filesystem::path powershell =
            std::filesystem::path(systemDirectory.data()) /
            "WindowsPowerShell/v1.0/powershell.exe";
        if (!std::filesystem::is_regular_file(powershell))
        {
            return GpuInstallResult::failed;
        }

        const std::wstring script =
            L"$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';"
            L"Expand-Archive -LiteralPath " +
            quotePowerShellLiteral(zipPath.wstring()) +
            L" -DestinationPath " + quotePowerShellLiteral(stagePath.wstring()) +
            L" -Force;"
            L"$provider=Get-ChildItem -LiteralPath " +
            quotePowerShellLiteral(stagePath.wstring()) +
            L" -Recurse -File -Filter 'onnxruntime_providers_cuda.dll'|Select-Object -First 1;"
            L"if($null -eq $provider){throw 'CUDA provider DLL was not found in the package'};"
            L"$gpu=Join-Path " + quotePowerShellLiteral(runtimeRoot.wstring()) +
            L" 'gpu';"
            L"New-Item -ItemType Directory -Path $gpu -Force|Out-Null;"
            L"Get-ChildItem -LiteralPath $provider.DirectoryName -Force|Copy-Item -Destination $gpu -Recurse -Force;";
        std::wstring commandLine =
            quoteCommandArgument(powershell.wstring()) +
            L" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command " +
            quoteCommandArgument(script);
        std::vector<wchar_t> mutableCommand(commandLine.begin(), commandLine.end());
        mutableCommand.push_back(L'\0');

        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESHOWWINDOW;
        startup.wShowWindow = SW_HIDE;
        PROCESS_INFORMATION process{};
        if (!CreateProcessW(
                powershell.c_str(),
                mutableCommand.data(),
                nullptr,
                nullptr,
                FALSE,
                CREATE_NO_WINDOW,
                nullptr,
                runtimeRoot.c_str(),
                &startup,
                &process))
        {
            return GpuInstallResult::failed;
        }
        CloseHandle(process.hThread);
        bool terminatedForCancel = false;
        while (WaitForSingleObject(process.hProcess, 200) == WAIT_TIMEOUT)
        {
            if (cancelRequested.load())
            {
                TerminateProcess(process.hProcess, 2);
                terminatedForCancel = true;
                WaitForSingleObject(process.hProcess, 5000);
                break;
            }
        }
        DWORD exitCode = 1;
        GetExitCodeProcess(process.hProcess, &exitCode);
        CloseHandle(process.hProcess);
        std::error_code cleanupError;
        std::filesystem::remove(zipPath, cleanupError);
        std::filesystem::remove_all(stagePath, cleanupError);
        if (terminatedForCancel || cancelRequested.load())
        {
            return GpuInstallResult::cancelled;
        }
        const bool installed =
            exitCode == 0 && std::filesystem::is_regular_file(
                                    runtimeRoot / "gpu/onnxruntime_providers_cuda.dll");
        if (installed)
        {
            PostMessageW(notifyWindow, WM_APP_GPU_INSTALL_PROGRESS, 100, 0);
        }
        return installed ? GpuInstallResult::succeeded
                         : GpuInstallResult::failed;
    }

    class MicrophoneRecorder
    {
    public:
        ~MicrophoneRecorder()
        {
            stop();
        }

        bool start(
            const std::filesystem::path &ffmpegPath,
            const std::wstring &deviceName,
            const std::filesystem::path &outputPath)
        {
            stop();
            SECURITY_ATTRIBUTES security{
                sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            HANDLE stdinRead = nullptr;
            HANDLE stdinWrite = nullptr;
            if (!CreatePipe(&stdinRead, &stdinWrite, &security, 0))
            {
                return false;
            }
            SetHandleInformation(stdinWrite, HANDLE_FLAG_INHERIT, 0);

            HANDLE nullOutput = CreateFileW(
                L"NUL",
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                &security,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (nullOutput == INVALID_HANDLE_VALUE)
            {
                CloseHandle(stdinRead);
                CloseHandle(stdinWrite);
                return false;
            }

            const std::wstring input = L"audio=" + deviceName;
            std::wstring commandLine =
                quoteCommandArgument(ffmpegPath.wstring()) +
                L" -y -hide_banner -loglevel error -f dshow -i " +
                quoteCommandArgument(input) +
                L" -vn -c:a aac -b:a 192k " +
                quoteCommandArgument(outputPath.wstring());
            std::vector<wchar_t> mutableCommand(
                commandLine.begin(), commandLine.end());
            mutableCommand.push_back(L'\0');

            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            startup.dwFlags = STARTF_USESTDHANDLES;
            startup.hStdInput = stdinRead;
            startup.hStdOutput = nullOutput;
            startup.hStdError = nullOutput;
            PROCESS_INFORMATION process{};
            const BOOL started = CreateProcessW(
                ffmpegPath.c_str(),
                mutableCommand.data(),
                nullptr,
                nullptr,
                TRUE,
                CREATE_NO_WINDOW,
                nullptr,
                ffmpegPath.parent_path().c_str(),
                &startup,
                &process);
            CloseHandle(stdinRead);
            CloseHandle(nullOutput);
            if (!started)
            {
                CloseHandle(stdinWrite);
                return false;
            }

            CloseHandle(process.hThread);
            process_ = process.hProcess;
            stdinWrite_ = stdinWrite;
            outputPath_ = outputPath;
            return true;
        }

        bool stop()
        {
            if (process_ == nullptr)
            {
                return true;
            }
            if (stdinWrite_ != nullptr)
            {
                constexpr char quit[]{'q', '\n'};
                DWORD written = 0;
                WriteFile(stdinWrite_, quit, sizeof(quit), &written, nullptr);
                CloseHandle(stdinWrite_);
                stdinWrite_ = nullptr;
            }

            bool forced = false;
            if (WaitForSingleObject(process_, 10000) == WAIT_TIMEOUT)
            {
                forced = true;
                TerminateProcess(process_, 1);
                WaitForSingleObject(process_, 2000);
            }
            DWORD exitCode = 1;
            GetExitCodeProcess(process_, &exitCode);
            CloseHandle(process_);
            process_ = nullptr;
            const bool success =
                !forced && exitCode == 0 &&
                std::filesystem::is_regular_file(outputPath_);
            outputPath_.clear();
            return success;
        }

    private:
        HANDLE process_{};
        HANDLE stdinWrite_{};
        std::filesystem::path outputPath_;
    };

    std::wstring getWindowTextString(HWND window)
    {
        const int length = GetWindowTextLengthW(window);
        std::wstring value(static_cast<std::size_t>(length + 1), L'\0');
        if (length > 0)
        {
            GetWindowTextW(window, value.data(), length + 1);
        }
        value.resize(static_cast<std::size_t>(length));
        return value;
    }

    void setFont(HWND control, HFONT font)
    {
        SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    }

    HWND makeControl(
        AppState &state,
        const wchar_t *className,
        const wchar_t *text,
        DWORD style,
        int x,
        int y,
        int width,
        int height,
        int id)
    {
        HWND control = CreateWindowExW(
            0,
            className,
            text,
            WS_CHILD | WS_VISIBLE | style,
            x,
            y,
            width,
            height,
            state.window,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
            GetModuleHandleW(nullptr),
            nullptr);
        setFont(control, state.font);
        return control;
    }

    void setStatus(AppState &state, const std::wstring &message)
    {
        SetWindowTextW(state.statusLabel, message.c_str());
    }

    std::filesystem::path settingsFilePath(const AppState &state)
    {
        std::vector<wchar_t> localAppData(32768, L'\0');
        const DWORD length = GetEnvironmentVariableW(
            L"LOCALAPPDATA",
            localAppData.data(),
            static_cast<DWORD>(localAppData.size()));
        if (length > 0 && length < localAppData.size())
        {
            return std::filesystem::path(localAppData.data()) /
                   "PetPetFace/settings.ini";
        }
        return state.runtimeRoot / "settings.ini";
    }

    void loadSettings(AppState &state)
    {
        const std::filesystem::path path = settingsFilePath(state);
        std::vector<wchar_t> value(32768, L'\0');
        GetPrivateProfileStringW(
            L"Devices",
            L"Camera",
            L"",
            value.data(),
            static_cast<DWORD>(value.size()),
            path.c_str());
        state.selectedCameraName = value.data();
        std::fill(value.begin(), value.end(), L'\0');
        GetPrivateProfileStringW(
            L"Devices",
            L"Microphone",
            DEFAULT_MICROPHONE_SETTING,
            value.data(),
            static_cast<DWORD>(value.size()),
            path.c_str());
        state.selectedMicrophoneName = value.data();
        if (state.selectedMicrophoneName == NO_MICROPHONE_SETTING)
        {
            state.selectedMicrophoneName.clear();
        }
        else if (state.selectedMicrophoneName.empty())
        {
            // Migrate the old empty/default setting to the system default device.
            state.selectedMicrophoneName = DEFAULT_MICROPHONE_SETTING;
        }

        state.cameraHotkey.virtualKey = static_cast<UINT>(GetPrivateProfileIntW(
            L"Hotkeys", L"CameraKey", 'Q', path.c_str()));
        state.cameraHotkey.modifiers = static_cast<BYTE>(GetPrivateProfileIntW(
            L"Hotkeys", L"CameraModifiers", 0, path.c_str()));
        state.recordingHotkey.virtualKey = static_cast<UINT>(GetPrivateProfileIntW(
            L"Hotkeys", L"RecordingKey", 'R', path.c_str()));
        state.recordingHotkey.modifiers = static_cast<BYTE>(GetPrivateProfileIntW(
            L"Hotkeys", L"RecordingModifiers", 0, path.c_str()));
        if (state.cameraHotkey.virtualKey == 0)
        {
            state.cameraHotkey = HotkeyBinding{'Q', 0};
        }
        if (state.recordingHotkey.virtualKey == 0)
        {
            state.recordingHotkey = HotkeyBinding{'R', 0};
        }
    }

    void saveSettings(const AppState &state)
    {
        const std::filesystem::path path = settingsFilePath(state);
        std::error_code directoryError;
        std::filesystem::create_directories(path.parent_path(), directoryError);
        WritePrivateProfileStringW(
            L"Devices", L"Camera", state.selectedCameraName.c_str(), path.c_str());
        const std::wstring microphoneSetting =
            state.selectedMicrophoneName.empty()
                ? NO_MICROPHONE_SETTING
                : state.selectedMicrophoneName;
        WritePrivateProfileStringW(
            L"Devices",
            L"Microphone",
            microphoneSetting.c_str(),
            path.c_str());
        const std::wstring cameraKey =
            std::to_wstring(state.cameraHotkey.virtualKey);
        const std::wstring cameraModifiers =
            std::to_wstring(state.cameraHotkey.modifiers);
        const std::wstring recordingKey =
            std::to_wstring(state.recordingHotkey.virtualKey);
        const std::wstring recordingModifiers =
            std::to_wstring(state.recordingHotkey.modifiers);
        WritePrivateProfileStringW(
            L"Hotkeys", L"CameraKey", cameraKey.c_str(), path.c_str());
        WritePrivateProfileStringW(
            L"Hotkeys",
            L"CameraModifiers",
            cameraModifiers.c_str(),
            path.c_str());
        WritePrivateProfileStringW(
            L"Hotkeys", L"RecordingKey", recordingKey.c_str(), path.c_str());
        WritePrivateProfileStringW(
            L"Hotkeys",
            L"RecordingModifiers",
            recordingModifiers.c_str(),
            path.c_str());
    }

    std::wstring formatHotkey(const HotkeyBinding &hotkey)
    {
        std::wstring text;
        if ((hotkey.modifiers & HOTKEYF_CONTROL) != 0)
        {
            text += L"Ctrl+";
        }
        if ((hotkey.modifiers & HOTKEYF_SHIFT) != 0)
        {
            text += L"Shift+";
        }
        if ((hotkey.modifiers & HOTKEYF_ALT) != 0)
        {
            text += L"Alt+";
        }

        if ((hotkey.virtualKey >= 'A' && hotkey.virtualKey <= 'Z') ||
            (hotkey.virtualKey >= '0' && hotkey.virtualKey <= '9'))
        {
            text.push_back(static_cast<wchar_t>(hotkey.virtualKey));
            return text;
        }
        if (hotkey.virtualKey >= VK_F1 && hotkey.virtualKey <= VK_F24)
        {
            text += L"F" + std::to_wstring(hotkey.virtualKey - VK_F1 + 1);
            return text;
        }

        UINT scanCode = MapVirtualKeyW(hotkey.virtualKey, MAPVK_VK_TO_VSC) << 16;
        if ((hotkey.modifiers & HOTKEYF_EXT) != 0)
        {
            scanCode |= 1U << 24;
        }
        wchar_t keyName[64]{};
        if (GetKeyNameTextW(static_cast<LONG>(scanCode), keyName, 64) > 0)
        {
            text += keyName;
        }
        else
        {
            text += L"VK " + std::to_wstring(hotkey.virtualKey);
        }
        return text;
    }

    void updateCameraButtonText(AppState &state, bool running)
    {
        const std::wstring text =
            (running ? L"关闭摄像头 (" : L"打开摄像头 (") +
            formatHotkey(state.cameraHotkey) + L")";
        SetWindowTextW(state.cameraButton, text.c_str());
    }

    bool matchesHotkey(const MSG &message, const HotkeyBinding &hotkey)
    {
        if ((message.message != WM_KEYDOWN && message.message != WM_SYSKEYDOWN) ||
            message.wParam != hotkey.virtualKey ||
            (message.lParam & (1LL << 30)) != 0)
        {
            return false;
        }
        BYTE modifiers = 0;
        if ((GetKeyState(VK_SHIFT) & 0x8000) != 0)
        {
            modifiers |= HOTKEYF_SHIFT;
        }
        if ((GetKeyState(VK_CONTROL) & 0x8000) != 0)
        {
            modifiers |= HOTKEYF_CONTROL;
        }
        if ((GetKeyState(VK_MENU) & 0x8000) != 0)
        {
            modifiers |= HOTKEYF_ALT;
        }
        constexpr BYTE modifierMask =
            HOTKEYF_SHIFT | HOTKEYF_CONTROL | HOTKEYF_ALT;
        return (modifiers & modifierMask) == (hotkey.modifiers & modifierMask);
    }

    std::wstring formatDuration(double totalSeconds)
    {
        if (!std::isfinite(totalSeconds) || totalSeconds < 0.0)
        {
            return L"--:--";
        }
        const int seconds = static_cast<int>(std::lround(totalSeconds));
        const int hours = seconds / 3600;
        const int minutes = (seconds % 3600) / 60;
        const int remainder = seconds % 60;
        std::wostringstream stream;
        if (hours > 0)
        {
            stream << std::setfill(L'0') << std::setw(2) << hours << L":";
        }
        stream << std::setfill(L'0') << std::setw(2) << minutes
               << L":" << std::setw(2) << remainder;
        return stream.str();
    }

    void updatePreview(AppState &state, const cv::Mat &frame)
    {
        std::lock_guard<std::mutex> lock(state.previewMutex);
        state.previewFrame = frame.clone();
    }

    void clearPreview(AppState &state)
    {
        {
            std::lock_guard<std::mutex> lock(state.previewMutex);
            state.previewFrame.release();
        }
        InvalidateRect(state.window, &state.previewRect, FALSE);
    }

    void drawPreview(AppState &state, HDC deviceContext)
    {
        const int availableWidth = state.previewRect.right - state.previewRect.left;
        const int availableHeight = state.previewRect.bottom - state.previewRect.top;
        if (availableWidth <= 1 || availableHeight <= 1)
        {
            return;
        }
        HDC memoryContext = CreateCompatibleDC(deviceContext);
        HBITMAP memoryBitmap =
            CreateCompatibleBitmap(deviceContext, availableWidth, availableHeight);
        HGDIOBJ oldBitmap = SelectObject(memoryContext, memoryBitmap);
        RECT localRect{0, 0, availableWidth, availableHeight};
        HBRUSH background = CreateSolidBrush(RGB(10, 13, 18));
        FillRect(memoryContext, &localRect, background);
        DeleteObject(background);

        HPEN borderPen = CreatePen(PS_SOLID, 1, THEME_BORDER);
        HGDIOBJ oldPen = SelectObject(memoryContext, borderPen);
        HGDIOBJ oldBrush = SelectObject(memoryContext, GetStockObject(HOLLOW_BRUSH));
        RoundRect(
            memoryContext,
            0,
            0,
            availableWidth,
            availableHeight,
            18,
            18);
        SelectObject(memoryContext, oldBrush);
        SelectObject(memoryContext, oldPen);
        DeleteObject(borderPen);

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(state.previewMutex);
            if (!state.previewFrame.empty())
            {
                frame = state.previewFrame.clone();
            }
        }
        if (frame.empty())
        {
            SetBkMode(memoryContext, TRANSPARENT);
            SetTextColor(memoryContext, THEME_MUTED);
            RECT textRect = localRect;
            DrawTextW(
                memoryContext,
                L"摄像头 / 视频预览",
                -1,
                &textRect,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }
        else
        {
            cv::Mat bgra;
            cv::cvtColor(frame, bgra, cv::COLOR_BGR2BGRA);
            const double scale = std::min(
                static_cast<double>(availableWidth) / bgra.cols,
                static_cast<double>(availableHeight) / bgra.rows);
            const int drawWidth = static_cast<int>(std::lround(bgra.cols * scale));
            const int drawHeight = static_cast<int>(std::lround(bgra.rows * scale));
            const int left = (availableWidth - drawWidth) / 2;
            const int top = (availableHeight - drawHeight) / 2;

            BITMAPINFO bitmapInfo{};
            bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bitmapInfo.bmiHeader.biWidth = bgra.cols;
            bitmapInfo.bmiHeader.biHeight = -bgra.rows;
            bitmapInfo.bmiHeader.biPlanes = 1;
            bitmapInfo.bmiHeader.biBitCount = 32;
            bitmapInfo.bmiHeader.biCompression = BI_RGB;
            SetStretchBltMode(memoryContext, HALFTONE);
            StretchDIBits(
                memoryContext,
                left,
                top,
                drawWidth,
                drawHeight,
                0,
                0,
                bgra.cols,
                bgra.rows,
                bgra.data,
                &bitmapInfo,
                DIB_RGB_COLORS,
                SRCCOPY);
        }

        BitBlt(
            deviceContext,
            state.previewRect.left,
            state.previewRect.top,
            availableWidth,
            availableHeight,
            memoryContext,
            0,
            0,
            SRCCOPY);
        SelectObject(memoryContext, oldBitmap);
        DeleteObject(memoryBitmap);
        DeleteDC(memoryContext);
    }

    void drawModernButton(const DRAWITEMSTRUCT &item)
    {
        COLORREF fill = THEME_CONTROL;
        if (item.CtlID == ID_PROCESS)
        {
            fill = THEME_ACCENT;
        }
        else if (item.CtlID == ID_CAMERA)
        {
            fill = THEME_CAMERA;
        }
        if ((item.itemState & ODS_DISABLED) != 0)
        {
            fill = RGB(60, 64, 75);
        }
        else if ((item.itemState & ODS_SELECTED) != 0)
        {
            fill = RGB(
                GetRValue(fill) * 4 / 5,
                GetGValue(fill) * 4 / 5,
                GetBValue(fill) * 4 / 5);
        }

        HBRUSH brush = CreateSolidBrush(fill);
        HPEN pen = CreatePen(PS_SOLID, 1, fill);
        HGDIOBJ oldBrush = SelectObject(item.hDC, brush);
        HGDIOBJ oldPen = SelectObject(item.hDC, pen);
        RoundRect(
            item.hDC,
            item.rcItem.left,
            item.rcItem.top,
            item.rcItem.right,
            item.rcItem.bottom,
            12,
            12);
        SelectObject(item.hDC, oldBrush);
        SelectObject(item.hDC, oldPen);
        DeleteObject(brush);
        DeleteObject(pen);

        wchar_t text[128]{};
        GetWindowTextW(item.hwndItem, text, static_cast<int>(std::size(text)));
        SetBkMode(item.hDC, TRANSPARENT);
        SetTextColor(item.hDC, THEME_TEXT);
        DrawTextW(
            item.hDC,
            text,
            -1,
            const_cast<RECT *>(&item.rcItem),
            DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    }

    bool chooseOpenVideo(HWND owner, std::wstring &path)
    {
        std::vector<wchar_t> buffer(32768, L'\0');
        OPENFILENAMEW dialog{};
        dialog.lStructSize = sizeof(dialog);
        dialog.hwndOwner = owner;
        dialog.lpstrFilter =
            L"视频文件\0*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v\0所有文件\0*.*\0";
        dialog.lpstrFile = buffer.data();
        dialog.nMaxFile = static_cast<DWORD>(buffer.size());
        dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
        if (!GetOpenFileNameW(&dialog))
        {
            return false;
        }
        path = buffer.data();
        return true;
    }

    bool setInputVideo(AppState &state, const std::filesystem::path &path)
    {
        if (!std::filesystem::is_regular_file(path))
        {
            setStatus(state, L"拖入的路径不是有效视频文件");
            return false;
        }
        std::wstring extension = path.extension().wstring();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](wchar_t character)
            { return static_cast<wchar_t>(towlower(character)); });
        const std::vector<std::wstring> supported{
            L".mp4", L".avi", L".mov", L".mkv", L".webm", L".m4v"};
        if (std::find(supported.begin(), supported.end(), extension) == supported.end())
        {
            setStatus(state, L"不支持该文件格式");
            return false;
        }

        state.inputPath = path;
        SetWindowTextW(state.inputEdit, path.filename().c_str());
        const std::filesystem::path output =
            path.parent_path() / (path.stem().wstring() + L"_petpet.mp4");
        SetWindowTextW(state.outputEdit, output.c_str());
        setStatus(state, L"已选择视频: " + path.filename().wstring());
        return true;
    }

    bool chooseOutputVideo(HWND owner, std::wstring &path)
    {
        std::vector<wchar_t> buffer(32768, L'\0');
        if (!path.empty())
        {
            wcsncpy_s(buffer.data(), buffer.size(), path.c_str(), _TRUNCATE);
        }
        OPENFILENAMEW dialog{};
        dialog.lStructSize = sizeof(dialog);
        dialog.hwndOwner = owner;
        dialog.lpstrFilter = L"MP4 视频\0*.mp4\0AVI 视频\0*.avi\0";
        dialog.lpstrFile = buffer.data();
        dialog.nMaxFile = static_cast<DWORD>(buffer.size());
        dialog.lpstrDefExt = L"mp4";
        dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
        if (!GetSaveFileNameW(&dialog))
        {
            return false;
        }
        path = buffer.data();
        return true;
    }

    int selectedTargetFps(const AppState &state)
    {
        const int selection = static_cast<int>(
            SendMessageW(state.fpsCombo, CB_GETCURSEL, 0, 0));
        constexpr int values[]{15, 24, 30, 60};
        if (selection < 0 || selection >= static_cast<int>(std::size(values)))
        {
            return 30;
        }
        return values[selection];
    }

    void refreshDeviceState(AppState &state)
    {
        const std::vector<std::wstring> cameras =
            enumerateDeviceNames(CLSID_VideoInputDeviceCategory);
        const std::vector<std::wstring> microphones =
            enumerateDeviceNames(CLSID_AudioInputDeviceCategory);
        if (cameras.empty())
        {
            state.selectedCameraIndex = 0;
            state.selectedCameraName.clear();
            EnableWindow(state.cameraButton, FALSE);
            setStatus(state, L"未检测到摄像头，请检查权限或设备占用");
            return;
        }

        auto camera = std::find(
            cameras.begin(), cameras.end(), state.selectedCameraName);
        if (camera == cameras.end())
        {
            state.selectedCameraIndex = 0;
            state.selectedCameraName = cameras.front();
        }
        else
        {
            state.selectedCameraIndex = static_cast<int>(
                std::distance(cameras.begin(), camera));
        }

        if (state.selectedMicrophoneName == DEFAULT_MICROPHONE_SETTING)
        {
            state.selectedMicrophoneName =
                microphones.empty() ? std::wstring{} : microphones.front();
        }
        else if (!state.selectedMicrophoneName.empty() &&
                 std::find(
                     microphones.begin(),
                     microphones.end(),
                     state.selectedMicrophoneName) == microphones.end())
        {
            state.selectedMicrophoneName =
                microphones.empty() ? std::wstring{} : microphones.front();
        }
        EnableWindow(state.cameraButton, TRUE);
        std::wostringstream status;
        status << L"设备就绪 · 摄像头: " << state.selectedCameraName
               << L" · 麦克风: "
               << (state.selectedMicrophoneName.empty()
                       ? L"不录音"
                       : state.selectedMicrophoneName);
        setStatus(state, status.str());
    }

    struct SettingsWindowState
    {
        AppState *app{};
        HWND window{};
        HWND cameraCombo{};
        HWND microphoneCombo{};
        HWND cameraHotkey{};
        HWND recordingHotkey{};
        HWND gpuDownloadButton{};
        HWND gpuProgressBar{};
        HWND gpuProgressLabel{};
        bool gpuInstallRunning{false};
        std::atomic_bool gpuCancelRequested{false};
        bool closeAfterGpuCancel{false};
        std::vector<std::wstring> cameraNames;
        std::vector<std::wstring> microphoneNames;
    };

    HWND makeSettingsControl(
        SettingsWindowState &settings,
        const wchar_t *className,
        const wchar_t *text,
        DWORD style,
        int x,
        int y,
        int width,
        int height,
        int id)
    {
        HWND control = CreateWindowExW(
            0,
            className,
            text,
            WS_CHILD | WS_VISIBLE | style,
            x,
            y,
            width,
            height,
            settings.window,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
            GetModuleHandleW(nullptr),
            nullptr);
        setFont(control, settings.app->font);
        return control;
    }

    void populateSettingsDevices(SettingsWindowState &settings)
    {
        settings.cameraNames =
            enumerateDeviceNames(CLSID_VideoInputDeviceCategory);
        settings.microphoneNames =
            enumerateDeviceNames(CLSID_AudioInputDeviceCategory);

        SendMessageW(settings.cameraCombo, CB_RESETCONTENT, 0, 0);
        if (settings.cameraNames.empty())
        {
            SendMessageW(
                settings.cameraCombo,
                CB_ADDSTRING,
                0,
                reinterpret_cast<LPARAM>(L"未检测到摄像头"));
            SendMessageW(settings.cameraCombo, CB_SETCURSEL, 0, 0);
        }
        else
        {
            int selected = 0;
            for (std::size_t index = 0; index < settings.cameraNames.size(); ++index)
            {
                const std::wstring &name = settings.cameraNames[index];
                SendMessageW(
                    settings.cameraCombo,
                    CB_ADDSTRING,
                    0,
                    reinterpret_cast<LPARAM>(name.c_str()));
                if (name == settings.app->selectedCameraName)
                {
                    selected = static_cast<int>(index);
                }
            }
            SendMessageW(settings.cameraCombo, CB_SETCURSEL, selected, 0);
        }

        SendMessageW(settings.microphoneCombo, CB_RESETCONTENT, 0, 0);
        SendMessageW(
            settings.microphoneCombo,
            CB_ADDSTRING,
            0,
            reinterpret_cast<LPARAM>(L"不录音"));
        int selectedMicrophone = 0;
        for (std::size_t index = 0; index < settings.microphoneNames.size(); ++index)
        {
            const std::wstring &name = settings.microphoneNames[index];
            SendMessageW(
                settings.microphoneCombo,
                CB_ADDSTRING,
                0,
                reinterpret_cast<LPARAM>(name.c_str()));
            if (name == settings.app->selectedMicrophoneName)
            {
                selectedMicrophone = static_cast<int>(index + 1);
            }
        }
        SendMessageW(
            settings.microphoneCombo,
            CB_SETCURSEL,
            selectedMicrophone,
            0);
    }

    LRESULT CALLBACK settingsWindowProcedure(
        HWND window, UINT message, WPARAM wParam, LPARAM lParam)
    {
        auto *settings = reinterpret_cast<SettingsWindowState *>(
            GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCCREATE)
        {
            auto *create = reinterpret_cast<CREATESTRUCTW *>(lParam);
            settings = static_cast<SettingsWindowState *>(create->lpCreateParams);
            settings->window = window;
            SetWindowLongPtrW(
                window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(settings));
        }
        if (settings == nullptr)
        {
            return DefWindowProcW(window, message, wParam, lParam);
        }

        switch (message)
        {
        case WM_CREATE:
            makeSettingsControl(
                *settings, L"STATIC", L"摄像头", 0, 24, 20, 430, 24, 0);
            settings->cameraCombo = makeSettingsControl(
                *settings,
                WC_COMBOBOXW,
                L"",
                CBS_DROPDOWNLIST | WS_VSCROLL,
                24,
                48,
                430,
                180,
                ID_SETTINGS_CAMERA);
            makeSettingsControl(
                *settings, L"STATIC", L"麦克风", 0, 24, 94, 430, 24, 0);
            settings->microphoneCombo = makeSettingsControl(
                *settings,
                WC_COMBOBOXW,
                L"",
                CBS_DROPDOWNLIST | WS_VSCROLL,
                24,
                118,
                430,
                180,
                ID_SETTINGS_MICROPHONE);
            makeSettingsControl(
                *settings,
                L"STATIC",
                L"打开 / 关闭摄像头快捷键",
                0,
                24,
                164,
                220,
                24,
                0);
            settings->cameraHotkey = makeSettingsControl(
                *settings,
                HOTKEY_CLASSW,
                L"",
                WS_BORDER,
                24,
                190,
                200,
                34,
                ID_SETTINGS_CAMERA_HOTKEY);
            SendMessageW(
                settings->cameraHotkey,
                HKM_SETHOTKEY,
                MAKEWORD(
                    settings->app->cameraHotkey.virtualKey,
                    settings->app->cameraHotkey.modifiers),
                0);
            makeSettingsControl(
                *settings,
                L"STATIC",
                L"开始 / 停止录制快捷键",
                0,
                254,
                164,
                200,
                24,
                0);
            settings->recordingHotkey = makeSettingsControl(
                *settings,
                HOTKEY_CLASSW,
                L"",
                WS_BORDER,
                254,
                190,
                200,
                34,
                ID_SETTINGS_RECORDING_HOTKEY);
            SendMessageW(
                settings->recordingHotkey,
                HKM_SETHOTKEY,
                MAKEWORD(
                    settings->app->recordingHotkey.virtualKey,
                    settings->app->recordingHotkey.modifiers),
                0);
            makeSettingsControl(
                *settings,
                L"BUTTON",
                L"刷新设备",
                BS_PUSHBUTTON,
                24,
                250,
                126,
                36,
                ID_SETTINGS_REFRESH);
            settings->gpuDownloadButton = makeSettingsControl(
                *settings,
                L"BUTTON",
                L"下载 GPU 组件",
                BS_PUSHBUTTON,
                254,
                250,
                200,
                36,
                ID_SETTINGS_GPU_DOWNLOAD);
            settings->gpuProgressBar = makeSettingsControl(
                *settings,
                PROGRESS_CLASSW,
                L"",
                PBS_SMOOTH,
                24,
                294,
                360,
                18,
                ID_SETTINGS_GPU_PROGRESS);
            SendMessageW(settings->gpuProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
            settings->gpuProgressLabel = makeSettingsControl(
                *settings,
                L"STATIC",
                L"0%",
                SS_RIGHT,
                392,
                290,
                62,
                24,
                0);
            ShowWindow(settings->gpuProgressBar, SW_HIDE);
            ShowWindow(settings->gpuProgressLabel, SW_HIDE);
            makeSettingsControl(
                *settings,
                L"BUTTON",
                L"保存",
                BS_DEFPUSHBUTTON,
                254,
                330,
                96,
                36,
                ID_SETTINGS_SAVE);
            makeSettingsControl(
                *settings,
                L"BUTTON",
                L"取消",
                BS_PUSHBUTTON,
                358,
                330,
                96,
                36,
                ID_SETTINGS_CANCEL);
            populateSettingsDevices(*settings);
            return 0;

        case WM_APP_GPU_INSTALL_PROGRESS:
        {
            const int progress = std::clamp(static_cast<int>(wParam), 0, 100);
            SendMessageW(settings->gpuProgressBar, PBM_SETPOS, progress, 0);
            const std::wstring label =
                progress >= 92 && progress < 100
                    ? L"解压中"
                    : std::to_wstring(progress) + L"%";
            SetWindowTextW(settings->gpuProgressLabel, label.c_str());
            return 0;
        }

        case WM_APP_GPU_INSTALL_COMPLETE:
        {
            const GpuInstallResult result =
                static_cast<GpuInstallResult>(wParam);
            settings->gpuInstallRunning = false;
            settings->gpuCancelRequested.store(false);
            EnableWindow(settings->gpuDownloadButton, TRUE);
            SetWindowTextW(settings->gpuDownloadButton, L"下载 GPU 组件");
            if (settings->closeAfterGpuCancel)
            {
                DestroyWindow(window);
                return 0;
            }
            MessageBoxW(
                window,
                result == GpuInstallResult::succeeded
                    ? L"GPU 组件已下载并解压到程序目录的 gpu 文件夹。请重启程序后勾选“使用 GPU”。"
                    : result == GpuInstallResult::cancelled
                          ? L"GPU 组件下载已取消，临时文件已清理。"
                          : L"GPU 组件下载或解压失败，请检查 HTTPS 网络连接和磁盘空间后重试。",
                L"GPU 组件",
                result == GpuInstallResult::failed
                    ? MB_ICONERROR
                    : MB_ICONINFORMATION);
            return 0;
        }

        case WM_COMMAND:
            switch (LOWORD(wParam))
            {
            case ID_SETTINGS_REFRESH:
                populateSettingsDevices(*settings);
                return 0;
            case ID_SETTINGS_GPU_DOWNLOAD:
                if (!settings->gpuInstallRunning)
                {
                    settings->gpuInstallRunning = true;
                    settings->gpuCancelRequested.store(false);
                    settings->closeAfterGpuCancel = false;
                    SendMessageW(settings->gpuProgressBar, PBM_SETPOS, 0, 0);
                    SetWindowTextW(settings->gpuProgressLabel, L"0%");
                    ShowWindow(settings->gpuProgressBar, SW_SHOW);
                    ShowWindow(settings->gpuProgressLabel, SW_SHOW);
                    SetWindowTextW(settings->gpuDownloadButton, L"停止下载");
                    const HWND notifyWindow = window;
                    const std::filesystem::path runtimeRoot =
                        settings->app->runtimeRoot;
                    std::atomic_bool *cancelRequested =
                        &settings->gpuCancelRequested;
                    std::thread([notifyWindow, runtimeRoot, cancelRequested]() {
                        const GpuInstallResult result = installGpuComponents(
                            runtimeRoot,
                            *cancelRequested,
                            notifyWindow);
                        PostMessageW(
                            notifyWindow,
                            WM_APP_GPU_INSTALL_COMPLETE,
                            static_cast<WPARAM>(result),
                            0);
                    }).detach();
                }
                else if (!settings->gpuCancelRequested.load())
                {
                    settings->gpuCancelRequested.store(true);
                    EnableWindow(settings->gpuDownloadButton, FALSE);
                    SetWindowTextW(settings->gpuDownloadButton, L"正在停止...");
                }
                return 0;
            case ID_SETTINGS_SAVE:
            {
                if (settings->gpuInstallRunning)
                {
                    MessageBoxW(
                        window,
                        L"GPU 组件仍在下载，请等待完成。",
                        L"GPU 组件",
                        MB_ICONINFORMATION);
                    return 0;
                }
                const WORD cameraHotkey = static_cast<WORD>(
                    SendMessageW(settings->cameraHotkey, HKM_GETHOTKEY, 0, 0));
                const WORD recordingHotkey = static_cast<WORD>(
                    SendMessageW(settings->recordingHotkey, HKM_GETHOTKEY, 0, 0));
                const HotkeyBinding cameraBinding{
                    LOBYTE(cameraHotkey), HIBYTE(cameraHotkey)};
                const HotkeyBinding recordingBinding{
                    LOBYTE(recordingHotkey), HIBYTE(recordingHotkey)};
                if (cameraBinding.virtualKey == 0 ||
                    recordingBinding.virtualKey == 0)
                {
                    MessageBoxW(
                        window,
                        L"两个快捷键都不能为空",
                        L"设备设置",
                        MB_ICONWARNING);
                    return 0;
                }
                if (cameraBinding.virtualKey == recordingBinding.virtualKey &&
                    cameraBinding.modifiers == recordingBinding.modifiers)
                {
                    MessageBoxW(
                        window,
                        L"摄像头和录制快捷键不能相同",
                        L"设备设置",
                        MB_ICONWARNING);
                    return 0;
                }
                if ((cameraBinding.virtualKey == VK_F4 &&
                     (cameraBinding.modifiers & HOTKEYF_ALT) != 0) ||
                    (recordingBinding.virtualKey == VK_F4 &&
                     (recordingBinding.modifiers & HOTKEYF_ALT) != 0))
                {
                    MessageBoxW(
                        window,
                        L"Alt+F4 是系统关闭窗口快捷键，请改用其他组合",
                        L"设备设置",
                        MB_ICONWARNING);
                    return 0;
                }
                const int cameraSelection = static_cast<int>(
                    SendMessageW(settings->cameraCombo, CB_GETCURSEL, 0, 0));
                if (!settings->cameraNames.empty() && cameraSelection >= 0)
                {
                    settings->app->selectedCameraIndex = cameraSelection;
                    settings->app->selectedCameraName =
                        settings->cameraNames[static_cast<std::size_t>(cameraSelection)];
                }
                const int microphoneSelection = static_cast<int>(
                    SendMessageW(
                        settings->microphoneCombo, CB_GETCURSEL, 0, 0));
                settings->app->selectedMicrophoneName =
                    microphoneSelection > 0
                        ? settings->microphoneNames[
                              static_cast<std::size_t>(microphoneSelection - 1)]
                        : std::wstring{};
                settings->app->cameraHotkey = cameraBinding;
                settings->app->recordingHotkey = recordingBinding;
                saveSettings(*settings->app);
                updateCameraButtonText(*settings->app, false);
                DestroyWindow(window);
                return 0;
            }
            case ID_SETTINGS_CANCEL:
            case IDCANCEL:
                if (settings->gpuInstallRunning)
                {
                    settings->closeAfterGpuCancel = true;
                    if (!settings->gpuCancelRequested.load())
                    {
                        settings->gpuCancelRequested.store(true);
                        EnableWindow(settings->gpuDownloadButton, FALSE);
                        SetWindowTextW(
                            settings->gpuDownloadButton, L"正在停止...");
                    }
                    return 0;
                }
                DestroyWindow(window);
                return 0;
            }
            break;

        case WM_CLOSE:
            if (settings->gpuInstallRunning)
            {
                settings->closeAfterGpuCancel = true;
                if (!settings->gpuCancelRequested.load())
                {
                    settings->gpuCancelRequested.store(true);
                    EnableWindow(settings->gpuDownloadButton, FALSE);
                    SetWindowTextW(
                        settings->gpuDownloadButton, L"正在停止...");
                }
                return 0;
            }
            DestroyWindow(window);
            return 0;

        case WM_DESTROY:
            settings->window = nullptr;
            return 0;
        }
        return DefWindowProcW(window, message, wParam, lParam);
    }

    void showSettings(AppState &state)
    {
        if (state.cameraRunning.load())
        {
            setStatus(state, L"请先关闭摄像头，再修改设备设置");
            return;
        }

        constexpr const wchar_t *className = L"PetPetFaceSettingsWindow";
        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.lpfnWndProc = settingsWindowProcedure;
        windowClass.hInstance = GetModuleHandleW(nullptr);
        windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        windowClass.lpszClassName = className;
        if (!RegisterClassExW(&windowClass) &&
            GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        {
            setStatus(state, L"无法打开设置窗口");
            return;
        }

        SettingsWindowState settings{&state};
        RECT owner{};
        GetWindowRect(state.window, &owner);
        const int width = 500;
        const int height = 440;
        HWND window = CreateWindowExW(
            WS_EX_DLGMODALFRAME,
            className,
            L"设备设置",
            WS_CAPTION | WS_SYSMENU,
            owner.left + ((owner.right - owner.left) - width) / 2,
            owner.top + ((owner.bottom - owner.top) - height) / 2,
            width,
            height,
            state.window,
            nullptr,
            GetModuleHandleW(nullptr),
            &settings);
        if (window == nullptr)
        {
            setStatus(state, L"无法打开设置窗口");
            return;
        }

        EnableWindow(state.window, FALSE);
        ShowWindow(window, SW_SHOW);
        MSG message{};
        while (settings.window != nullptr)
        {
            const BOOL result = GetMessageW(&message, nullptr, 0, 0);
            if (result <= 0)
            {
                if (result == 0)
                {
                    PostQuitMessage(static_cast<int>(message.wParam));
                }
                break;
            }
            if (!IsDialogMessageW(window, &message))
            {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        EnableWindow(state.window, TRUE);
        SetForegroundWindow(state.window);
        refreshDeviceState(state);
    }

    void stopCamera(AppState &state)
    {
        state.recordingRequested.store(false);
        state.cameraRunning.store(false);
        if (state.cameraThread.joinable())
        {
            state.cameraThread.join();
        }
        state.actualFps.store(0.0);
        clearPreview(state);
        updateCameraButtonText(state, false);
    }

    void cameraWorker(
        AppState *state,
        int cameraIndex,
        bool useCuda,
        std::wstring microphoneName)
    {
        cv::VideoWriter recordingWriter;
        std::filesystem::path recordingPath;
        std::filesystem::path recordingVideoPath;
        std::filesystem::path recordingAudioPath;
        MicrophoneRecorder microphoneRecorder;
        bool recordingSession = false;
        bool recordingHasAudio = false;
        auto finishRecording = [&]()
        {
            if (!recordingSession)
            {
                return;
            }
            recordingSession = false;
            recordingWriter.release();
            state->recordingActive.store(false);
            bool success = true;
            std::wstring message;
            try
            {
                if (recordingHasAudio)
                {
                    if (!microphoneRecorder.stop())
                    {
                        throw std::runtime_error("Microphone recording failed");
                    }
                    petpet_face::mergeOriginalAudio(
                        recordingAudioPath,
                        recordingVideoPath,
                        recordingPath);
                    std::error_code removeError;
                    std::filesystem::remove(recordingVideoPath, removeError);
                    std::filesystem::remove(recordingAudioPath, removeError);
                }
                message = L"录制已保存: " + recordingPath.wstring();
            }
            catch (const std::exception &error)
            {
                success = false;
                microphoneRecorder.stop();
                std::error_code removeError;
                std::filesystem::remove(recordingVideoPath, removeError);
                std::filesystem::remove(recordingAudioPath, removeError);
                std::filesystem::remove(recordingPath, removeError);
                const std::wstring detail(
                    error.what(), error.what() + strlen(error.what()));
                message = L"录制失败: " + detail;
            }
            PostMessageW(
                state->window,
                WM_APP_RECORDING_STATE,
                0,
                reinterpret_cast<LPARAM>(new RecordingStateUpdate{
                    false,
                    success,
                    std::move(message)}));
        };
        try
        {
            petpet_face::DetectorConfig detectorConfig;
            detectorConfig.useCuda = useCuda;
            auto detector = std::make_unique<petpet_face::FaceDetector>(
                state->runtimeRoot / "models/face/face_yolo11s.onnx",
                detectorConfig);
            const std::wstring executionProvider(
                detector->executionProvider().begin(),
                detector->executionProvider().end());
            petpet_face::PetPetRenderer renderer(
                state->runtimeRoot / "assets/petpet_frames");
            petpet_face::FaceTracker tracker;

            struct DetectionPipe
            {
                std::mutex mutex;
                std::condition_variable ready;
                cv::Mat pendingFrame;
                bool hasPendingFrame{};
                bool stop{};
                std::vector<cv::Rect> latestBoxes;
                std::uint64_t version{};
                std::exception_ptr error;
            } pipe;

            auto detectionLoop = [&]()
            {
                try
                {
                    while (true)
                    {
                        cv::Mat detectionFrame;
                        {
                            std::unique_lock<std::mutex> lock(pipe.mutex);
                            pipe.ready.wait(lock, [&]()
                                            { return pipe.stop || pipe.hasPendingFrame; });
                            if (pipe.stop)
                            {
                                break;
                            }
                            detectionFrame = std::move(pipe.pendingFrame);
                            pipe.hasPendingFrame = false;
                        }
                        auto boxes = detector->detect(detectionFrame);
                        {
                            std::lock_guard<std::mutex> lock(pipe.mutex);
                            pipe.latestBoxes = std::move(boxes);
                            ++pipe.version;
                        }
                    }
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(pipe.mutex);
                    pipe.error = std::current_exception();
                    pipe.stop = true;
                }
            };
            std::thread detectionThread;

            cv::VideoCapture capture;

            const std::vector<int> captureParameters{
                cv::CAP_PROP_FRAME_WIDTH, 1280,
                cv::CAP_PROP_FRAME_HEIGHT, 720,
                cv::CAP_PROP_FPS, state->targetFps};

            // 固定使用 Media Foundation 720p，避免启动时逐级探测多个模式。
            if (!capture.open(
                    cameraIndex,
                    cv::CAP_MSMF,
                    captureParameters))
            {
                throw std::runtime_error("Could not open camera in 720p MSMF mode");
            }

            capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);

            if (!capture.isOpened())
            {
                throw std::runtime_error("Could not open camera");
            }

            detectionThread = std::thread(detectionLoop);
            struct DetectionThreadGuard
            {
                DetectionPipe &pipe;
                std::thread &thread;
                ~DetectionThreadGuard()
                {
                    {
                        std::lock_guard<std::mutex> lock(pipe.mutex);
                        pipe.stop = true;
                    }
                    pipe.ready.notify_one();
                    if (thread.joinable())
                    {
                        thread.join();
                    }
                }
            } detectionGuard{pipe, detectionThread};

            const int actualWidth = static_cast<int>(
                capture.get(cv::CAP_PROP_FRAME_WIDTH));
            const int actualHeight = static_cast<int>(
                capture.get(cv::CAP_PROP_FRAME_HEIGHT));
            const double cameraReportedFps =
                capture.get(cv::CAP_PROP_FPS);

            std::wostringstream runningStatus;
            runningStatus
                << L"摄像头运行中 · "
                << actualWidth << L"×" << actualHeight
                << L" · 报告 " << std::fixed << std::setprecision(1)
                << cameraReportedFps << L" FPS"
                << L" · MSMF 720p"
                << L" · " << executionProvider;

            PostMessageW(
                state->window,
                WM_APP_CAMERA_STATE,
                0,
                reinterpret_cast<LPARAM>(
                    new CameraStateUpdate{true, runningStatus.str()}));

            const auto startTime = std::chrono::steady_clock::now();
            auto fpsStart = startTime;
            int framesInWindow = 0;
            std::uint64_t consumedVersion = 0;
            std::vector<cv::Rect> stableFaces;
            cv::Mat frame;
            while (state->cameraRunning.load() && capture.read(frame) && !frame.empty())
            {
                const auto frameStart = std::chrono::steady_clock::now();
                cv::flip(frame, frame, 1);

                if (state->effectEnabled.load())
                {
                    bool submittedForDetection = false;
                    {
                        std::lock_guard<std::mutex> lock(pipe.mutex);
                        // 检测线程尚未取走上一帧时，不重复 clone 覆盖，
                        // 避免推理较慢时拖累摄像头显示帧率。
                        if (!pipe.hasPendingFrame)
                        {
                            pipe.pendingFrame = frame.clone();
                            pipe.hasPendingFrame = true;
                            submittedForDetection = true;
                        }
                    }
                    if (submittedForDetection)
                    {
                        pipe.ready.notify_one();
                    }

                    std::vector<cv::Rect> latestBoxes;
                    bool hasNewResult = false;
                    {
                        std::lock_guard<std::mutex> lock(pipe.mutex);
                        if (pipe.version != consumedVersion)
                        {
                            consumedVersion = pipe.version;
                            latestBoxes = pipe.latestBoxes;
                            hasNewResult = true;
                        }
                    }
                    if (hasNewResult)
                    {
                        stableFaces = tracker.update(latestBoxes, frame.size());
                    }
                }
                else
                {
                    stableFaces.clear();
                }

                const auto now = std::chrono::steady_clock::now();
                const auto elapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime)
                        .count();
                if (state->effectEnabled.load())
                {
                    renderer.render(frame, stableFaces, elapsedMs);
                }

                if (state->recordingRequested.load() && !recordingWriter.isOpened())
                {
                    const std::filesystem::path recordingDirectory =
                        state->runtimeRoot / "recordings";
                    std::filesystem::create_directories(recordingDirectory);
                    SYSTEMTIME localTime{};
                    GetLocalTime(&localTime);
                    std::wostringstream fileName;
                    fileName << L"petpet_"
                             << std::setfill(L'0')
                             << std::setw(4) << localTime.wYear
                             << std::setw(2) << localTime.wMonth
                             << std::setw(2) << localTime.wDay << L"_"
                             << std::setw(2) << localTime.wHour
                             << std::setw(2) << localTime.wMinute
                             << std::setw(2) << localTime.wSecond
                             << L".mp4";
                    recordingPath = recordingDirectory / fileName.str();
                    recordingHasAudio = !microphoneName.empty();
                    recordingVideoPath = recordingHasAudio
                                             ? petpet_face::makeVideoOnlyPath(recordingPath)
                                             : recordingPath;
                    recordingAudioPath = recordingHasAudio
                                             ? recordingDirectory /
                                                   (L"." + recordingPath.stem().wstring() +
                                                    L"_audio_only.m4a")
                                             : std::filesystem::path{};
                    recordingWriter.open(
                        recordingVideoPath.string(),
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        static_cast<double>(std::max(1, state->targetFps)),
                        frame.size());
                    bool audioStarted = true;
                    if (recordingWriter.isOpened() && recordingHasAudio)
                    {
                        const std::filesystem::path ffmpegPath =
                            state->runtimeRoot / "tools/ffmpeg.exe";
                        audioStarted = std::filesystem::is_regular_file(ffmpegPath) &&
                                       microphoneRecorder.start(
                                           ffmpegPath,
                                           microphoneName,
                                           recordingAudioPath);
                    }
                    if (!recordingWriter.isOpened() || !audioStarted)
                    {
                        recordingWriter.release();
                        microphoneRecorder.stop();
                        std::error_code removeError;
                        std::filesystem::remove(recordingVideoPath, removeError);
                        std::filesystem::remove(recordingAudioPath, removeError);
                        state->recordingRequested.store(false);
                        state->recordingActive.store(false);
                        PostMessageW(
                            state->window,
                            WM_APP_RECORDING_STATE,
                            0,
                            reinterpret_cast<LPARAM>(new RecordingStateUpdate{
                                false,
                                false,
                                audioStarted
                                    ? L"录制失败: 无法创建 MP4 文件"
                                    : L"录制失败: 无法打开所选麦克风"}));
                    }
                    else
                    {
                        recordingSession = true;
                        state->recordingActive.store(true);
                        PostMessageW(
                            state->window,
                            WM_APP_RECORDING_STATE,
                            0,
                            reinterpret_cast<LPARAM>(new RecordingStateUpdate{
                                true,
                                true,
                                (recordingHasAudio
                                     ? L"正在录制视频和麦克风 ("
                                     : L"正在录制无声视频 (") +
                                    formatHotkey(state->recordingHotkey) +
                                    L" 停止): " + recordingPath.wstring()}));
                    }
                }

                if (recordingWriter.isOpened())
                {
                    if (state->recordingRequested.load())
                    {
                        recordingWriter.write(frame);
                    }
                    else
                    {
                        finishRecording();
                    }
                }
                updatePreview(*state, frame);

                ++framesInWindow;
                const double fpsElapsed =
                    std::chrono::duration<double>(now - fpsStart).count();
                if (fpsElapsed >= 0.5)
                {
                    state->actualFps.store(framesInWindow / fpsElapsed);
                    framesInWindow = 0;
                    fpsStart = now;
                }

                const auto targetDuration = std::chrono::milliseconds(
                    std::max(1, 1000 / std::max(1, state->targetFps)));
                const auto used = std::chrono::steady_clock::now() - frameStart;
                if (used < targetDuration)
                {
                    std::this_thread::sleep_for(targetDuration - used);
                }
            }
            finishRecording();
            capture.release();
            {
                std::lock_guard<std::mutex> lock(pipe.mutex);
                pipe.stop = true;
            }
            pipe.ready.notify_one();
            detectionThread.join();
            if (pipe.error)
            {
                std::rethrow_exception(pipe.error);
            }
            state->cameraRunning.store(false);
            state->recordingRequested.store(false);
            state->recordingActive.store(false);
            clearPreview(*state);
            PostMessageW(
                state->window,
                WM_APP_CAMERA_STATE,
                0,
                reinterpret_cast<LPARAM>(
                    new CameraStateUpdate{false, L"摄像头已关闭"}));
        }
        catch (const std::exception &error)
        {
            finishRecording();
            state->cameraRunning.store(false);
            state->recordingRequested.store(false);
            state->recordingActive.store(false);
            clearPreview(*state);
            const std::wstring message(error.what(), error.what() + strlen(error.what()));
            PostMessageW(
                state->window,
                WM_APP_CAMERA_STATE,
                0,
                reinterpret_cast<LPARAM>(
                    new CameraStateUpdate{false, L"摄像头错误: " + message}));
        }
    }

    void toggleCamera(AppState &state)
    {
        if (state.processing.load())
        {
            setStatus(state, L"视频处理期间不能打开摄像头");
            return;
        }
        if (state.cameraRunning.load())
        {
            stopCamera(state);
            setStatus(state, L"摄像头已关闭");
            return;
        }
        if (state.cameraThread.joinable())
        {
            state.cameraThread.join();
        }
        state.targetFps = selectedTargetFps(state);
        state.recordingRequested.store(false);
        state.recordingActive.store(false);
        const bool useCuda =
            SendMessageW(state.gpuCheck, BM_GETCHECK, 0, 0) == BST_CHECKED;
        state.cameraRunning.store(true);
        EnableWindow(GetDlgItem(state.window, ID_SETTINGS), FALSE);
        updateCameraButtonText(state, true);
        setStatus(state, L"正在启动摄像头...");
        const int cameraIndex = state.selectedCameraIndex;
        const std::wstring microphoneName = state.selectedMicrophoneName;
        state.cameraThread =
            std::thread(
                cameraWorker,
                &state,
                cameraIndex,
                useCuda,
                microphoneName);
    }

    void toggleRecording(AppState &state)
    {
        if (state.processing.load())
        {
            setStatus(state, L"视频处理期间不能录制摄像头");
            return;
        }
        if (!state.cameraRunning.load())
        {
            setStatus(
                state,
                L"请先按 " + formatHotkey(state.cameraHotkey) +
                    L" 打开摄像头，再按 " +
                    formatHotkey(state.recordingHotkey) + L" 开始录制");
            return;
        }
        if (state.recordingRequested.load())
        {
            state.recordingRequested.store(false);
            setStatus(state, L"正在停止录制...");
        }
        else
        {
            state.recordingRequested.store(true);
            setStatus(state, L"正在开始录制...");
        }
    }

    void videoWorker(
        AppState *state,
        std::filesystem::path inputPath,
        std::filesystem::path outputPath,
        bool useCuda)
    {
        std::filesystem::path videoOnlyPath;
        try
        {
            petpet_face::DetectorConfig detectorConfig;
            detectorConfig.useCuda = useCuda;
            petpet_face::FaceDetector detector(
                state->runtimeRoot / "models/face/face_yolo11s.onnx",
                detectorConfig);
            const std::wstring executionProvider(
                detector.executionProvider().begin(),
                detector.executionProvider().end());
            petpet_face::PetPetRenderer renderer(
                state->runtimeRoot / "assets/petpet_frames");
            petpet_face::FaceTracker tracker;

            cv::VideoCapture capture(inputPath.string());
            if (!capture.isOpened())
            {
                throw std::runtime_error("Could not open input video");
            }
            const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
            const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
            const int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
            double sourceFps = capture.get(cv::CAP_PROP_FPS);
            if (sourceFps <= 0.0)
            {
                sourceFps = 25.0;
            }
            std::filesystem::create_directories(outputPath.parent_path());
            videoOnlyPath = petpet_face::makeVideoOnlyPath(outputPath);
            cv::VideoWriter writer(
                videoOnlyPath.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                sourceFps,
                cv::Size(width, height));
            if (!writer.isOpened())
            {
                throw std::runtime_error("Could not create output video");
            }

            const auto startTime = std::chrono::steady_clock::now();
            int completed = 0;
            cv::Mat frame;
            while (!state->cancelProcessing.load() &&
                   capture.read(frame) && !frame.empty())
            {
                const auto faces = detector.detect(frame);
                const auto stableFaces = tracker.update(faces, frame.size());
                const auto elapsedMs = static_cast<std::int64_t>(
                    std::llround(completed * 1000.0 / sourceFps));
                if (state->effectEnabled.load())
                {
                    renderer.render(frame, stableFaces, elapsedMs);
                }
                writer.write(frame);
                ++completed;
                if (completed % 2 == 0)
                {
                    updatePreview(*state, frame);
                }

                if (completed % 3 == 0 || completed == totalFrames)
                {
                    const double elapsed = std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - startTime)
                                               .count();
                    const double processingFps = completed > 0 ? completed / elapsed : 0.0;
                    const double remaining =
                        processingFps > 0.0 && totalFrames > completed
                            ? (totalFrames - completed) / processingFps
                            : 0.0;
                    PostMessageW(
                        state->window,
                        WM_APP_PROGRESS,
                        0,
                        reinterpret_cast<LPARAM>(new ProgressUpdate{
                            completed, totalFrames, processingFps, remaining}));
                }
            }
            writer.release();
            capture.release();
            if (state->cancelProcessing.load())
            {
                std::error_code removeError;
                std::filesystem::remove(videoOnlyPath, removeError);
                throw std::runtime_error("Processing cancelled");
            }
            setStatus(*state, L"正在合并原视频音频...");
            petpet_face::mergeOriginalAudio(inputPath, videoOnlyPath, outputPath);
            {
                std::error_code removeError;
                std::filesystem::remove(videoOnlyPath, removeError);
            }
            state->processing.store(false);
            PostMessageW(
                state->window,
                WM_APP_COMPLETE,
                0,
                reinterpret_cast<LPARAM>(new CompletionUpdate{
                    true,
                    L"处理完成 · 已保留原音频 · " + executionProvider}));
        }
        catch (const std::exception &error)
        {
            if (!videoOnlyPath.empty())
            {
                std::error_code removeError;
                std::filesystem::remove(videoOnlyPath, removeError);
            }
            state->processing.store(false);
            const std::wstring message(error.what(), error.what() + strlen(error.what()));
            PostMessageW(
                state->window,
                WM_APP_COMPLETE,
                0,
                reinterpret_cast<LPARAM>(new CompletionUpdate{
                    false, L"处理失败: " + message}));
        }
    }

    void startVideoProcessing(AppState &state)
    {
        if (state.processing.load())
        {
            return;
        }
        stopCamera(state);
        if (state.processingThread.joinable())
        {
            state.processingThread.join();
        }
        const std::filesystem::path inputPath = state.inputPath;
        const std::filesystem::path outputPath(getWindowTextString(state.outputEdit));
        if (!std::filesystem::is_regular_file(inputPath) || outputPath.empty())
        {
            setStatus(state, L"请选择有效的输入视频和输出路径");
            return;
        }
        const bool useCuda =
            SendMessageW(state.gpuCheck, BM_GETCHECK, 0, 0) == BST_CHECKED;
        state.cancelProcessing.store(false);
        state.processing.store(true);
        SendMessageW(state.progressBar, PBM_SETPOS, 0, 0);
        EnableWindow(state.processButton, FALSE);
        setStatus(state, L"正在处理视频...");
        state.processingThread =
            std::thread(videoWorker, &state, inputPath, outputPath, useCuda);
    }

    void createInterface(AppState &state)
    {
        state.font = CreateFontW(
            -18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        state.titleFont = CreateFontW(
            -30, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        state.smallFont = CreateFontW(
            -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        state.backgroundBrush = CreateSolidBrush(THEME_BACKGROUND);
        state.cardBrush = CreateSolidBrush(THEME_CARD);
        state.controlBrush = CreateSolidBrush(THEME_CONTROL);

        HWND title = makeControl(
            state, L"STATIC", L"PetPet Face Studio", SS_LEFT,
            24, 16, 210, 38, 0);
        setFont(title, state.titleFont);
        HWND subtitle = makeControl(
            state, L"STATIC", L"实时人脸摸摸头与视频处理", SS_LEFT,
            26, 52, 210, 22, 0);
        setFont(subtitle, state.smallFont);
        makeControl(
            state, L"BUTTON", L"设置", BS_OWNERDRAW,
            248, 20, 88, 38, ID_SETTINGS);

        makeControl(state, L"STATIC", L"输入视频", 0, 28, 88, 300, 24, 0);
        state.inputEdit = makeControl(
            state,
            L"BUTTON",
            L"在此处拖入或者点击选择视频",
            BS_OWNERDRAW,
            28,
            114,
            308,
            36,
            ID_INPUT_BROWSE);

        makeControl(state, L"STATIC", L"输出视频", 0, 28, 158, 300, 24, 0);
        state.outputEdit = makeControl(
            state, L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL,
            28, 184, 218, 32, ID_OUTPUT_EDIT);
        makeControl(
            state, L"BUTTON", L"浏览", BS_OWNERDRAW,
            254, 184, 82, 32, ID_OUTPUT_BROWSE);

        state.processButton = makeControl(
            state, L"BUTTON", L"开始处理视频", BS_OWNERDRAW,
            28, 234, 308, 40, ID_PROCESS);
        state.cameraButton = makeControl(
            state, L"BUTTON", L"打开摄像头 (Q)", BS_OWNERDRAW,
            28, 282, 308, 40, ID_CAMERA);

        state.effectCheck = makeControl(
            state, L"BUTTON", L"启用摸摸头特效", BS_AUTOCHECKBOX,
            28, 340, 188, 28, ID_EFFECT);
        SendMessageW(state.effectCheck, BM_SETCHECK, BST_CHECKED, 0);
        state.gpuCheck = makeControl(
            state, L"BUTTON", L"使用 GPU", BS_AUTOCHECKBOX,
            224, 340, 112, 28, ID_GPU);

        makeControl(state, L"STATIC", L"摄像头目标 FPS", 0, 28, 388, 166, 26, 0);
        state.fpsCombo = makeControl(
            state, WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_VSCROLL,
            202, 384, 134, 150, ID_TARGET_FPS);
        for (const wchar_t *value : {L"15", L"24", L"30", L"60"})
        {
            SendMessageW(state.fpsCombo, CB_ADDSTRING, 0,
                         reinterpret_cast<LPARAM>(value));
        }
        SendMessageW(state.fpsCombo, CB_SETCURSEL, 2, 0);

        state.actualFpsLabel = makeControl(
            state, L"STATIC", L"实际 FPS: 0.0", 0,
            28, 428, 308, 26, 0);

        makeControl(state, L"STATIC", L"视频处理进度", 0, 28, 462, 308, 24, 0);
        state.progressBar = makeControl(
            state, PROGRESS_CLASSW, L"", PBS_SMOOTH,
            28, 490, 308, 18, ID_PROGRESS);
        SendMessageW(state.progressBar, PBM_SETRANGE32, 0, 1000);
        SendMessageW(state.progressBar, PBM_SETBARCOLOR, 0, THEME_ACCENT);
        SendMessageW(state.progressBar, PBM_SETBKCOLOR, 0, THEME_CONTROL);
        state.progressLabel = makeControl(
            state, L"STATIC", L"0 / 0 帧 (0%)", 0,
            28, 518, 308, 24, 0);
        state.remainingLabel = makeControl(
            state, L"STATIC", L"预计剩余: --:--", 0,
            28, 546, 308, 24, 0);
        state.statusLabel = makeControl(
            state,
            L"STATIC",
            L"就绪",
            SS_LEFTNOWORDWRAP | SS_CENTERIMAGE | SS_ENDELLIPSIS,
            28, 684, 1062, 34, 0);
        refreshDeviceState(state);
        updateCameraButtonText(state, false);
    }

    void layoutInterface(AppState &state)
    {
        RECT client{};
        GetClientRect(state.window, &client);
        const int clientWidth = client.right - client.left;
        const int clientHeight = client.bottom - client.top;
        const int statusTop = std::max(684, clientHeight - 38);

        state.previewRect = RECT{
            366,
            82,
            std::max(367, clientWidth - 14),
            std::max(83, statusTop - 34)};
        state.sidebarBottom = std::max(658, statusTop - 26);

        MoveWindow(
            state.statusLabel,
            28,
            statusTop,
            std::max(1, clientWidth - 42),
            34,
            TRUE);
    }

    LRESULT CALLBACK windowProcedure(
        HWND window, UINT message, WPARAM wParam, LPARAM lParam)
    {
        AppState *state = reinterpret_cast<AppState *>(
            GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCCREATE)
        {
            auto *create = reinterpret_cast<CREATESTRUCTW *>(lParam);
            state = static_cast<AppState *>(create->lpCreateParams);
            state->window = window;
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
        }
        if (state == nullptr)
        {
            return DefWindowProcW(window, message, wParam, lParam);
        }

        switch (message)
        {
        case WM_CREATE:
        {
            const BOOL darkMode = TRUE;
            DwmSetWindowAttribute(
                window, 20, &darkMode, sizeof(darkMode));
            createInterface(*state);
            layoutInterface(*state);
            DragAcceptFiles(window, TRUE);
            SetTimer(window, TIMER_REFRESH, 33, nullptr);
            return 0;
        }

        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED)
            {
                layoutInterface(*state);
                InvalidateRect(window, nullptr, TRUE);
            }
            return 0;

        case WM_GETMINMAXINFO:
        {
            auto *minimum = reinterpret_cast<MINMAXINFO *>(lParam);
            minimum->ptMinTrackSize.x = 1120;
            minimum->ptMinTrackSize.y = 760;
            return 0;
        }

        case WM_COMMAND:
            switch (LOWORD(wParam))
            {
            case ID_INPUT_BROWSE:
            {
                std::wstring path;
                if (chooseOpenVideo(window, path))
                {
                    setInputVideo(*state, std::filesystem::path(path));
                }
                return 0;
            }
            case ID_OUTPUT_BROWSE:
            {
                std::wstring path = getWindowTextString(state->outputEdit);
                if (chooseOutputVideo(window, path))
                {
                    SetWindowTextW(state->outputEdit, path.c_str());
                }
                return 0;
            }
            case ID_PROCESS:
                startVideoProcessing(*state);
                return 0;
            case ID_CAMERA:
                toggleCamera(*state);
                return 0;
            case ID_SETTINGS:
                showSettings(*state);
                return 0;
            case ID_EFFECT:
                state->effectEnabled.store(
                    SendMessageW(state->effectCheck, BM_GETCHECK, 0, 0) == BST_CHECKED);
                setStatus(
                    *state,
                    state->effectEnabled.load() ? L"特效已开启" : L"特效已关闭");
                return 0;
            }
            break;

        case WM_TIMER:
            if (wParam == TIMER_REFRESH)
            {
                std::wostringstream fps;
                fps << L"实际 FPS: " << std::fixed << std::setprecision(1)
                    << state->actualFps.load();
                SetWindowTextW(state->actualFpsLabel, fps.str().c_str());
                InvalidateRect(window, &state->previewRect, FALSE);
            }
            return 0;

        case WM_DROPFILES:
        {
            const HDROP drop = reinterpret_cast<HDROP>(wParam);
            std::vector<wchar_t> path(32768, L'\0');
            if (DragQueryFileW(
                    drop,
                    0,
                    path.data(),
                    static_cast<UINT>(path.size())) > 0)
            {
                setInputVideo(*state, std::filesystem::path(path.data()));
            }
            DragFinish(drop);
            return 0;
        }

        case WM_DRAWITEM:
            drawModernButton(*reinterpret_cast<DRAWITEMSTRUCT *>(lParam));
            return TRUE;

        case WM_CTLCOLORSTATIC:
        {
            HDC deviceContext = reinterpret_cast<HDC>(wParam);
            SetBkMode(deviceContext, OPAQUE);
            SetBkColor(deviceContext, THEME_CARD);
            SetTextColor(deviceContext, THEME_TEXT);
            return reinterpret_cast<LRESULT>(state->cardBrush);
        }

        case WM_CTLCOLOREDIT:
        case WM_CTLCOLORLISTBOX:
        {
            HDC deviceContext = reinterpret_cast<HDC>(wParam);
            SetBkColor(deviceContext, THEME_CONTROL);
            SetTextColor(deviceContext, THEME_TEXT);
            return reinterpret_cast<LRESULT>(state->controlBrush);
        }

        case WM_CTLCOLORBTN:
        {
            HDC deviceContext = reinterpret_cast<HDC>(wParam);
            SetBkMode(deviceContext, TRANSPARENT);
            SetTextColor(deviceContext, THEME_TEXT);
            return reinterpret_cast<LRESULT>(state->cardBrush);
        }

        case WM_APP_PROGRESS:
        {
            std::unique_ptr<ProgressUpdate> update(
                reinterpret_cast<ProgressUpdate *>(lParam));
            const int progress = update->total > 0
                                     ? static_cast<int>(1000.0 * update->completed / update->total)
                                     : 0;
            SendMessageW(state->progressBar, PBM_SETPOS, progress, 0);
            std::wostringstream label;
            label << update->completed << L" / " << update->total << L" 帧 ("
                  << progress / 10 << L"%)  处理 FPS: "
                  << std::fixed << std::setprecision(1) << update->fps;
            SetWindowTextW(state->progressLabel, label.str().c_str());
            const std::wstring remaining =
                L"预计剩余: " + formatDuration(update->remainingSeconds);
            SetWindowTextW(state->remainingLabel, remaining.c_str());
            return 0;
        }

        case WM_APP_COMPLETE:
        {
            std::unique_ptr<CompletionUpdate> update(
                reinterpret_cast<CompletionUpdate *>(lParam));
            if (state->processingThread.joinable())
            {
                state->processingThread.join();
            }
            EnableWindow(state->processButton, TRUE);
            setStatus(*state, update->message);
            if (!update->success)
            {
                MessageBoxW(window, update->message.c_str(), L"PetPet Face", MB_ICONERROR);
            }
            return 0;
        }

        case WM_APP_CAMERA_STATE:
        {
            std::unique_ptr<CameraStateUpdate> update(
                reinterpret_cast<CameraStateUpdate *>(lParam));
            if (!update->running)
            {
                updateCameraButtonText(*state, false);
                state->actualFps.store(0.0);
                EnableWindow(GetDlgItem(state->window, ID_SETTINGS), TRUE);
            }
            setStatus(*state, update->message);
            return 0;
        }

        case WM_APP_RECORDING_STATE:
        {
            std::unique_ptr<RecordingStateUpdate> update(
                reinterpret_cast<RecordingStateUpdate *>(lParam));
            state->recordingActive.store(update->active);
            setStatus(*state, update->message);
            if (!update->success)
            {
                MessageBoxW(window, update->message.c_str(), L"PetPet Face", MB_ICONERROR);
            }
            return 0;
        }

        case WM_PAINT:
        {
            PAINTSTRUCT paint{};
            HDC deviceContext = BeginPaint(window, &paint);
            const bool previewOnly =
                paint.rcPaint.left >= state->previewRect.left &&
                paint.rcPaint.top >= state->previewRect.top &&
                paint.rcPaint.right <= state->previewRect.right &&
                paint.rcPaint.bottom <= state->previewRect.bottom;
            if (!previewOnly)
            {
                FillRect(deviceContext, &paint.rcPaint, state->backgroundBrush);
            }

            HBRUSH cardBrush = CreateSolidBrush(THEME_CARD);
            HPEN cardPen = CreatePen(PS_SOLID, 1, THEME_BORDER);
            HGDIOBJ oldBrush = SelectObject(deviceContext, cardBrush);
            HGDIOBJ oldPen = SelectObject(deviceContext, cardPen);
            RoundRect(
                deviceContext,
                14,
                10,
                352,
                state->sidebarBottom,
                20,
                20);
            SelectObject(deviceContext, oldBrush);
            SelectObject(deviceContext, oldPen);
            DeleteObject(cardBrush);
            DeleteObject(cardPen);

            drawPreview(*state, deviceContext);
            EndPaint(window, &paint);
            return 0;
        }

        case WM_CLOSE:
            state->cancelProcessing.store(true);
            state->recordingRequested.store(false);
            state->cameraRunning.store(false);
            if (state->cameraThread.joinable())
            {
                state->cameraThread.join();
            }
            if (state->processingThread.joinable())
            {
                state->processingThread.join();
            }
            DestroyWindow(window);
            return 0;

        case WM_DESTROY:
            KillTimer(window, TIMER_REFRESH);
            if (state->font != nullptr)
            {
                DeleteObject(state->font);
                state->font = nullptr;
            }
            if (state->titleFont != nullptr)
            {
                DeleteObject(state->titleFont);
                state->titleFont = nullptr;
            }
            if (state->smallFont != nullptr)
            {
                DeleteObject(state->smallFont);
                state->smallFont = nullptr;
            }
            if (state->backgroundBrush != nullptr)
            {
                DeleteObject(state->backgroundBrush);
                state->backgroundBrush = nullptr;
            }
            if (state->controlBrush != nullptr)
            {
                DeleteObject(state->controlBrush);
                state->controlBrush = nullptr;
            }
            if (state->cardBrush != nullptr)
            {
                DeleteObject(state->cardBrush);
                state->cardBrush = nullptr;
            }
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcW(window, message, wParam, lParam);
    }

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    const HRESULT comResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    INITCOMMONCONTROLSEX commonControls{
        sizeof(INITCOMMONCONTROLSEX),
        ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES | ICC_WIN95_CLASSES};
    InitCommonControlsEx(&commonControls);

    AppState state;
    try
    {
        state.runtimeRoot = findRuntimeRoot();
        loadSettings(state);
    }
    catch (const std::exception &error)
    {
        const std::wstring message(error.what(), error.what() + strlen(error.what()));
        MessageBoxW(nullptr, message.c_str(), L"PetPet Face", MB_ICONERROR);
        return 1;
    }

    constexpr const wchar_t *className = L"PetPetFaceGuiWindow";
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = windowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    windowClass.lpszClassName = className;
    if (!RegisterClassExW(&windowClass))
    {
        return 1;
    }

    HWND window = CreateWindowExW(
        0,
        className,
        L"PetPet Face - 实时摸摸头",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1120,
        760,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 1;
    }
    ShowWindow(window, showCommand);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0)
    {
        const bool cameraHotkeyPressed =
            matchesHotkey(message, state.cameraHotkey);
        const bool recordingHotkeyPressed =
            matchesHotkey(message, state.recordingHotkey);
        if (cameraHotkeyPressed || recordingHotkeyPressed)
        {
            wchar_t focusedClass[32]{};
            GetClassNameW(GetFocus(), focusedClass, 32);
            if (_wcsicmp(focusedClass, L"Edit") != 0)
            {
                if (cameraHotkeyPressed)
                {
                    toggleCamera(state);
                }
                else
                {
                    toggleRecording(state);
                }
                continue;
            }
        }
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    if (SUCCEEDED(comResult))
    {
        CoUninitialize();
    }
    return static_cast<int>(message.wParam);
}
