cmake_minimum_required(VERSION 3.16)
project(PPEGloveDetection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)

add_executable(ppe_detector
    src/main.cpp
    src/detector.cpp
    src/utils.cpp
)

target_include_directories(ppe_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ppe_detector PRIVATE ${OpenCV_LIBS})

if (MSVC)
    target_compile_options(ppe_detector PRIVATE /W4)
else()
    target_compile_options(ppe_detector PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS ppe_detector RUNTIME DESTINATION bin)
