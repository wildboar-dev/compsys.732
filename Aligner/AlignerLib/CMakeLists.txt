#--------------------------------------------------------
# CMake for generating the AlignerLib
#
# @author: Wild Boar
#
# Date Created: 2024-07-29
#--------------------------------------------------------

# Setup include directories
include_directories("../" "${LIBRARY_BASE}/NVLib")

# Create Library
add_library (AlignerLib STATIC
	ArgUtils.cpp
	LoadUtils.cpp
	HelperUtils.cpp
	ICPEngine.cpp
	ColorMatcher.cpp
)

