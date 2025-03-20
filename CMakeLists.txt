cmake_minimum_required(VERSION 3.15...3.31)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# include(FetchContent)
# FetchContent_Declare(stim
#     GIT_REPOSITORY https://github.com/quantumlib/stim.git
#     GIT_TAG v1.14.0)
# FetchContent_GetProperties(stim)
# if(NOT stim_POPULATED)
#     FetchContent_Populate(stim)
#     add_subdirectory(${stim_SOURCE_DIR})
# endif()

# Core source files - utility and code implementation.
set(WEAVE_CORE_SOURCES
    weave/_core/src/codes/noise_model.cpp
    weave/_core/src/util/pcm.cpp
    weave/_core/src/util/graph.cpp
)

# Python binding source files.
set(WEAVE_BINDINGS_SOURCES
    weave/_core/src/bindings/weave.pybind.cpp
    weave/_core/src/bindings/codes.pybind.cpp
    weave/_core/src/bindings/util.pybind.cpp
)

# Create the Python module.
pybind11_add_module(_core ${WEAVE_CORE_SOURCES} ${WEAVE_BINDINGS_SOURCES})
target_include_directories(_core PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/weave/_core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/weave/_core/lib/eigen
)
# target_link_libraries(_core PRIVATE libstim)

# Install the module to the Python package directory.
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})

# Define header files for IDE integration.
file(GLOB_RECURSE WEAVE_HEADERS
    weave/_core/include/weave/*.hpp
    weave/_core/include/weave/*.h
    weave/_core/include/bindings/*.hpp
    weave/_core/include/bindings/*.h
)

# Add headers to target sources for IDE integration.
target_sources(_core PRIVATE ${WEAVE_HEADERS})
