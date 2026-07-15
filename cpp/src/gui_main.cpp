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
constexpr int ID_CAMERA_SELECT = 111;
constexpr int ID_CAMERA_REFRESH = 112;

constexpr UINT WM_APP_PROGRESS = WM_APP + 1;
constexpr UINT WM_APP_COMPLETE = WM_APP + 2;
constexpr UINT WM_APP_CAMERA_STATE = WM_APP + 3;
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

struct AppState
{
    HWND window{};
    HWND inputEdit{};
    HWND outputEdit{};
    HWND processButton{};
    HWND cameraButton{};
    HWND cameraCombo{};
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
    std::atomic_bool cameraRunning{false};
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

std::vector<std::wstring> enumerateCameraNames()
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
            CLSID_VideoInputDeviceCategory,
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
        [](wchar_t character) { return static_cast<wchar_t>(towlower(character)); });
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

void refreshCameraList(AppState &state)
{
    SendMessageW(state.cameraCombo, CB_RESETCONTENT, 0, 0);
    const std::vector<std::wstring> names = enumerateCameraNames();
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        std::wostringstream label;
        label << index << L"  ·  " << names[index];
        const std::wstring text = label.str();
        SendMessageW(
            state.cameraCombo,
            CB_ADDSTRING,
            0,
            reinterpret_cast<LPARAM>(text.c_str()));
    }
    if (names.empty())
    {
        SendMessageW(
            state.cameraCombo,
            CB_ADDSTRING,
            0,
            reinterpret_cast<LPARAM>(L"未检测到摄像头"));
        EnableWindow(state.cameraButton, FALSE);
        setStatus(state, L"DirectShow 未检测到摄像头，请检查权限或设备占用");
    }
    else
    {
        EnableWindow(state.cameraButton, TRUE);
        std::wostringstream status;
        status << L"已检测到 " << names.size() << L" 个摄像头";
        setStatus(state, status.str());
    }
    SendMessageW(state.cameraCombo, CB_SETCURSEL, 0, 0);
}

void stopCamera(AppState &state)
{
    state.cameraRunning.store(false);
    if (state.cameraThread.joinable())
    {
        state.cameraThread.join();
    }
    state.actualFps.store(0.0);
    clearPreview(state);
    SetWindowTextW(state.cameraButton, L"打开摄像头 (Q)");
}

void cameraWorker(AppState *state, int cameraIndex, bool useCuda)
{
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

        auto detectionLoop = [&]() {
            try
            {
                while (true)
                {
                    cv::Mat detectionFrame;
                    {
                        std::unique_lock<std::mutex> lock(pipe.mutex);
                        pipe.ready.wait(lock, [&]() {
                            return pipe.stop || pipe.hasPendingFrame;
                        });
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
        capture.open(cameraIndex, cv::CAP_DSHOW);
        if (!capture.isOpened())
        {
            capture.release();
            capture.open(cameraIndex);
        }
        if (!capture.isOpened())
        {
            throw std::runtime_error("Could not open camera");
        }
        capture.set(
            cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280.0);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720.0);
        capture.set(cv::CAP_PROP_FPS, state->targetFps);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);
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

        std::wostringstream runningStatus;
        runningStatus
            << L"摄像头运行中 · "
            << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)) << L"×"
            << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT))
            << L" · 请求 " << state->targetFps << L" FPS"
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
                {
                    std::lock_guard<std::mutex> lock(pipe.mutex);
                    pipe.pendingFrame = frame.clone();
                    pipe.hasPendingFrame = true;
                }
                pipe.ready.notify_one();

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
        state->cameraRunning.store(false);
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
    const bool useCuda =
        SendMessageW(state.gpuCheck, BM_GETCHECK, 0, 0) == BST_CHECKED;
    state.cameraRunning.store(true);
    SetWindowTextW(state.cameraButton, L"关闭摄像头 (Q)");
    setStatus(state, L"正在启动摄像头...");
    int cameraIndex = static_cast<int>(
        SendMessageW(state.cameraCombo, CB_GETCURSEL, 0, 0));
    if (cameraIndex < 0)
    {
        cameraIndex = 0;
    }
    state.cameraThread =
        std::thread(cameraWorker, &state, cameraIndex, useCuda);
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
                    std::chrono::steady_clock::now() - startTime).count();
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
        24, 16, 312, 38, 0);
    setFont(title, state.titleFont);
    HWND subtitle = makeControl(
        state, L"STATIC", L"实时人脸摸摸头与视频处理", SS_LEFT,
        26, 52, 310, 22, 0);
    setFont(subtitle, state.smallFont);

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

    makeControl(state, L"STATIC", L"摄像头设备", 0, 28, 382, 166, 26, 0);
    state.cameraCombo = makeControl(
        state, WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_VSCROLL,
        28, 408, 218, 180, ID_CAMERA_SELECT);
    makeControl(
        state, L"BUTTON", L"刷新", BS_OWNERDRAW,
        254, 408, 82, 32, ID_CAMERA_REFRESH);

    makeControl(state, L"STATIC", L"摄像头目标 FPS", 0, 28, 452, 166, 26, 0);
    state.fpsCombo = makeControl(
        state, WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_VSCROLL,
        202, 448, 134, 150, ID_TARGET_FPS);
    for (const wchar_t *value : {L"15", L"24", L"30", L"60"})
    {
        SendMessageW(state.fpsCombo, CB_ADDSTRING, 0,
                     reinterpret_cast<LPARAM>(value));
    }
    SendMessageW(state.fpsCombo, CB_SETCURSEL, 2, 0);

    state.actualFpsLabel = makeControl(
        state, L"STATIC", L"实际 FPS: 0.0", 0,
        28, 492, 308, 26, 0);

    makeControl(state, L"STATIC", L"视频处理进度", 0, 28, 526, 308, 24, 0);
    state.progressBar = makeControl(
        state, PROGRESS_CLASSW, L"", PBS_SMOOTH,
        28, 554, 308, 18, ID_PROGRESS);
    SendMessageW(state.progressBar, PBM_SETRANGE32, 0, 1000);
    SendMessageW(state.progressBar, PBM_SETBARCOLOR, 0, THEME_ACCENT);
    SendMessageW(state.progressBar, PBM_SETBKCOLOR, 0, THEME_CONTROL);
    state.progressLabel = makeControl(
        state, L"STATIC", L"0 / 0 帧 (0%)", 0,
        28, 582, 308, 24, 0);
    state.remainingLabel = makeControl(
        state, L"STATIC", L"预计剩余: --:--", 0,
        28, 610, 308, 24, 0);
    state.statusLabel = makeControl(
        state, L"STATIC", L"就绪", SS_LEFT,
        28, 684, 1062, 34, 0);
    refreshCameraList(state);
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
        case ID_EFFECT:
            state->effectEnabled.store(
                SendMessageW(state->effectCheck, BM_GETCHECK, 0, 0) == BST_CHECKED);
            setStatus(
                *state,
                state->effectEnabled.load() ? L"特效已开启" : L"特效已关闭");
            return 0;
        case ID_CAMERA_REFRESH:
            if (!state->cameraRunning.load())
            {
                refreshCameraList(*state);
            }
            else
            {
                setStatus(*state, L"请先关闭摄像头再刷新设备列表");
            }
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
            SetWindowTextW(state->cameraButton, L"打开摄像头 (Q)");
            state->actualFps.store(0.0);
        }
        setStatus(*state, update->message);
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
        sizeof(INITCOMMONCONTROLSEX), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&commonControls);

    AppState state;
    try
    {
        state.runtimeRoot = findRuntimeRoot();
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
        if (message.message == WM_KEYDOWN &&
            (message.wParam == 'Q' || message.wParam == 'q'))
        {
            wchar_t focusedClass[32]{};
            GetClassNameW(GetFocus(), focusedClass, 32);
            if (_wcsicmp(focusedClass, L"Edit") != 0)
            {
                toggleCamera(state);
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
