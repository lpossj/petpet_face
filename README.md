# PetPet Face

基于 C++17、YOLO11 ONNX、ONNX Runtime 和 OpenCV 的实时人脸摸摸头程序。

[查看更新日志](CHANGELOG.md)

<p align="center">
  <img src="assets/petpet.gif" alt="PetPet Face 演示效果" width="720">
</p>

## 功能

- 本地视频处理，默认保留原视频音频
- DirectShow 摄像头选择与实时特效
- 顶部“设置”窗口选择摄像头和麦克风；首次启动默认使用系统默认麦克风，也可选“不录音”
- 设置窗口可自定义摄像头和录制快捷键，组合键与设备选择自动保存
- 设置窗口提供“下载 GPU 组件”按钮；程序通过 HTTPS 流式下载并显示进度和百分比，可停止下载或直接关闭设置，完成后自动解压到运行目录的 `gpu/` 文件夹
- `R` 键同步录制摄像头特效画面和所选麦克风，再按一次停止
- CPU / NVIDIA CUDA 推理
- 人脸跟踪、平滑、短时丢失补偿
- 人脸顶部向下形变、动态羽化边缘
- 可点击或拖入视频的响应式 Win32 GUI

## 仓库结构

```text
assets/                 透明手部动画帧
cpp/include/            C++ 头文件
cpp/src/                C++ 源代码
models/face/            YOLO11 人脸 ONNX 模型
tools/                  FFmpeg 许可证；可自行放置 ffmpeg.exe
```

`cpp/build/`、输出视频、第三方 DLL 和 FFmpeg 可执行文件不提交 Git，发布版请从 GitHub Releases 下载。

## 构建

要求：Visual Studio C++、CMake、OpenCV 4.12、ONNX Runtime 1.25。

```powershell
$env:OpenCV_DIR = "<包含 OpenCVConfig.cmake 的目录>"
$env:ONNXRUNTIME_ROOT = "<ONNX Runtime 根目录>"

cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release --parallel
```

CUDA 构建可额外指定运行库目录：

```powershell
cmake -S cpp -B cpp/build `
  -DPETPET_CUDA_RUNTIME_DIR="<CUDA bin>" `
  -DPETPET_CUDNN_RUNTIME_DIR="<cuDNN bin>"
```

代码不保存机器绝对路径；这些路径只存在于本机 CMake 缓存。

## 运行

GUI：

```powershell
./cpp/build/Release/petpet_face_gui.exe
```

先在顶部“设置”中选择摄像头、麦克风和快捷键。默认快捷键：`Q` 打开或关闭摄像头；
摄像头开启后，`R` 开始或停止录制。可改为单键或 `Ctrl` / `Shift` / `Alt` 组合键。
视频与麦克风音频自动合并为 MP4，
保存到程序目录下的 `recordings/`。选择“不录音”时输出无声 MP4。

命令行：

```powershell
./cpp/build/Release/petpet_face.exe `
  ./models/face/face_yolo11s.onnx `
  ./assets/petpet_frames `
  ./input/test.mp4 `
  ./output/test_petpet.mp4 `
  --gpu
```

处理视频音频需要 `tools/ffmpeg.exe`。程序按相对路径自动查找。

## 模型说明

`models/face/face_yolo11s.onnx` 是本项目使用的人脸检测模型，
由 YOLO11s 模型训练并导出为 ONNX 格式。

模型用于本地人脸区域检测，不上传用户的视频、图片或摄像头数据。
