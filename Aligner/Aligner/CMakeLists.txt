#--------------------------------------------------------
# CMake for generating the main executable
#
# @author: Wild Boar
#
# Date Created: 2024-07-29
#--------------------------------------------------------

# Setup the includes
include_directories("../")

# Create the executable
add_executable(Aligner
    Engine.cpp
    Source.cpp
)

# Add link libraries                               
target_link_libraries(Aligner AlignerLib Jarvis NVLib ${OpenCV_LIBS} OpenSSL::SSL uuid)

# Copy Resources across
add_custom_target(resource_copy ALL
	COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/Resources ${CMAKE_BINARY_DIR}/Aligner 
    DEPENDS ${MY_TARGET}
)