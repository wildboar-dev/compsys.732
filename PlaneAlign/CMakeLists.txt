#--------------------------------------------------------
# Top-Level: PAlign
#
# @author: Wild Boar
#
# @Date Created: 2025-03-30
#--------------------------------------------------------

cmake_minimum_required(VERSION 3.0.0)
project(PAlign VERSION 0.1.0)

# Set the base path of the libraries
set(LIBRARY_BASE "$ENV{HOME}/Libraries")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add OpenCV to the solution
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Include OpenSSL
find_package(OpenSSL REQUIRED)

# Add the NVLib library to the folder
add_library(NVLib STATIC IMPORTED)
set_target_properties(NVLib PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_BASE}/NVLib/build/NVLib/libNVLib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_BASE}/NVLib"
)

# Create the executable
add_executable(PAlign
    Source.cpp
)

# Add link libraries                               
target_link_libraries(PAlign NVLib ${OpenCV_LIBS} OpenSSL::SSL uuid)

# Copy Resources across
add_custom_target(resource_copy ALL
	COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/Data ${CMAKE_BINARY_DIR}/
    DEPENDS ${MY_TARGET}
)

# Create an output folder
add_custom_target(output_copy ALL
	COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/Output ${CMAKE_BINARY_DIR}/Output
    DEPENDS ${MY_TARGET}
)