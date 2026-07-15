cmake_minimum_required(VERSION 3.20)
project(PetPetFace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
    file(TO_CMAKE_PATH "$ENV{OpenCV_DIR}" OpenCV_DIR)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn videoio highgui)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    file(TO_CMAKE_PATH "$ENV{ONNXRUNTIME_ROOT}" ONNXRUNTIME_ROOT)
endif()

set(PETPET_CUDA_RUNTIME_DIR "" CACHE PATH
    "Directory containing CUDA 12 runtime DLLs")
if(NOT PETPET_CUDA_RUNTIME_DIR AND DEFINED ENV{PETPET_CUDA_RUNTIME_DIR})
    file(TO_CMAKE_PATH "$ENV{PETPET_CUDA_RUNTIME_DIR}" PETPET_CUDA_RUNTIME_DIR)
endif()
set(PETPET_CUDNN_RUNTIME_DIR "" CACHE PATH
    "Directory containing cuDNN 9 runtime DLLs")
if(NOT PETPET_CUDNN_RUNTIME_DIR AND DEFINED ENV{PETPET_CUDNN_RUNTIME_DIR})
    file(TO_CMAKE_PATH "$ENV{PETPET_CUDNN_RUNTIME_DIR}" PETPET_CUDNN_RUNTIME_DIR)
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include build/native/include buildTransitive/native/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64 runtimes/win-x64/native)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT.")
endif()

add_library(petpet_face_core STATIC
    src/audio_muxer.cpp
    src/face_detector.cpp
    src/face_tracker.cpp
    src/image_preprocessor.cpp
    src/petpet_renderer.cpp
    src/media_processor.cpp)
target_include_directories(petpet_face_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(petpet_face_core PUBLIC
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIBRARY}")

if(MSVC)
    target_compile_options(petpet_face_core PRIVATE /utf-8 /W4)
else()
    target_compile_options(petpet_face_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(petpet_face src/main.cpp)
target_link_libraries(petpet_face PRIVATE petpet_face_core)

if(WIN32)
    add_executable(petpet_face_gui WIN32 src/gui_main.cpp)
    target_compile_definitions(petpet_face_gui PRIVATE UNICODE _UNICODE)
    target_link_libraries(petpet_face_gui PRIVATE
        petpet_face_core
        comctl32
        comdlg32
        dwmapi
        shell32
        ole32
        oleaut32
        strmiids)
    if(MSVC)
        target_compile_options(petpet_face_gui PRIVATE /utf-8 /W4)
    endif()
endif()

if(WIN32 AND (PETPET_CUDA_RUNTIME_DIR OR PETPET_CUDNN_RUNTIME_DIR))
    file(GLOB PETPET_CUDA_RUNTIME_DLLS CONFIGURE_DEPENDS
        "${PETPET_CUDA_RUNTIME_DIR}/cublas*.dll"
        "${PETPET_CUDA_RUNTIME_DIR}/cudart*.dll"
        "${PETPET_CUDA_RUNTIME_DIR}/nvrtc*.dll"
        "${PETPET_CUDA_RUNTIME_DIR}/zlibwapi.dll"
        "${PETPET_CUDNN_RUNTIME_DIR}/cudnn*.dll")
    if(NOT PETPET_CUDA_RUNTIME_DLLS)
        message(FATAL_ERROR
            "PETPET_CUDA_RUNTIME_DIR contains no CUDA/cuDNN runtime DLLs")
    endif()
    foreach(TARGET_NAME IN ITEMS petpet_face petpet_face_gui)
        if(TARGET ${TARGET_NAME})
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/gpu"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${PETPET_CUDA_RUNTIME_DLLS}
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/gpu"
                COMMAND_EXPAND_LISTS)
        endif()
    endforeach()
endif()

if(WIN32)
    get_filename_component(ONNXRUNTIME_NATIVE_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    file(GLOB ONNXRUNTIME_DLLS CONFIGURE_DEPENDS "${ONNXRUNTIME_NATIVE_DIR}/*.dll")
    foreach(DLL IN LISTS ONNXRUNTIME_DLLS)
        add_custom_command(TARGET petpet_face POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL}" "$<TARGET_FILE_DIR:petpet_face>")
        if(TARGET petpet_face_gui)
            add_custom_command(TARGET petpet_face_gui POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL}" "$<TARGET_FILE_DIR:petpet_face_gui>")
        endif()
    endforeach()
endif()
