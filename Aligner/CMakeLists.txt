#--------------------------------------------------------
# Top-Level: Aligner
#
# @author: Wild Boar
#
# @Date Created: 2024-07-29
#--------------------------------------------------------

# Create the main application
cmake_minimum_required(VERSION 3.5.0)
project(Aligner VERSION 0.1.0)

# Enable Testing
enable_testing()

# Set the correct version of C++
set(CMAKE_CXX_STANDARD 17)

# Setup base directory
set(LIBRARY_BASE $ENV{HOME}/Libraries)

# Setup the includes for the libraries
include_directories("../" "${LIBRARY_BASE}/NVLib" "${LIBRARY_BASE}/JarvisLib")

# Add the NVLib library to the folder
add_library(NVLib STATIC IMPORTED)
set_target_properties(NVLib PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_BASE}/NVLib/build/NVLib/libNVLib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_BASE}/NVLib"
)

# Add the NVLib library to the folder
add_library(Jarvis STATIC IMPORTED)
set_target_properties(Jarvis PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_BASE}/JarvisLib/build/JarvisLib/libJarvisLib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_BASE}/JarvisLib"
)

# Add OpenCV to the solution
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Include OpenSSL
find_package(OpenSSL REQUIRED)

# Add the available subfolders
add_subdirectory(AlignerLib)
add_subdirectory(AlignerTests)
add_subdirectory(Aligner)