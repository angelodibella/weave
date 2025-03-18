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

# Include directories.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/weave/_core/include)

# Core source files.
set(WEAVE_CORE_SOURCES
    weave/_core/src/codes/noise_model.cpp
    weave/_core/src/codes/hypergraph_product_code.cpp
    weave/_core/src/codes/stabilizer_code.cpp
    weave/_core/src/util/pcm.cpp
    weave/_core/src/util/graph.cpp
)

# Binding source files.
set(WEAVE_BINDINGS_SOURCES
    weave/_core/src/bindings/weave.pybind.cpp
)

# Create the Python module.
pybind11_add_module(_core ${WEAVE_CORE_SOURCES} ${WEAVE_BINDINGS_SOURCES})
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/weave/_core/include)

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