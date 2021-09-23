# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CVV_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

project (camera_calibration)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable (main src/main.cpp)
target_link_libraries (main ${OpenCV_LIBS})