cmake_minimum_required(VERSION 3.15...4.3)
project(options_pricing_engine LANGUAGES CXX)

# ─── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── pybind11 (modern FindPython mode for v3.0+) ─────────────────────────────
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# ─── Build the pricer Python module ──────────────────────────────────────────
pybind11_add_module(pricer src/bindings.cpp)

# Header-only pricer — include the src/ directory so #include "pricer/bs_pricer.hpp" works
target_include_directories(pricer PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ─── Compiler-specific optimisations ─────────────────────────────────────────
if(MSVC)
    # MSVC: max optimisation, fast float, multi-processor build, M_PI constants
    target_compile_options(pricer PRIVATE /O2 /fp:fast /MP /bigobj)
    target_compile_definitions(pricer PRIVATE _USE_MATH_DEFINES)
    target_link_libraries(pricer PRIVATE pybind11::windows_extras)
else()
    # GCC / Clang
    target_compile_options(pricer PRIVATE -O3 -march=native -ffast-math)
endif()

# ─── Output the .pyd / .so into the project root for easy import ─────────────
set_target_properties(pricer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"              # Windows .pyd
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
)
