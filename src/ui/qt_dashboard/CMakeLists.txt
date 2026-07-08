cmake_minimum_required(VERSION 3.10)
project(UavRadarDashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Network)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Network)
find_package(OpenCV REQUIRED)

add_executable(qt_dashboard
    main.cpp
    VideoWidget.cpp
    RadarWidget.cpp
)

target_link_libraries(qt_dashboard
    PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Network
    ${OpenCV_LIBS}
)

if(UNIX)
    target_link_libraries(qt_dashboard PRIVATE rt)
endif()

if(WIN32)
    target_link_libraries(qt_dashboard PRIVATE Psapi)
endif()
