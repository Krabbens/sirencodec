cmake_minimum_required(VERSION 3.24)

project(sirencodec_infer
  VERSION 0.1.0
  DESCRIPTION "C++20 SirenCodec TFLite/LiteRT inference runtime"
  LANGUAGES CXX)

option(SIRENCODEC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(SIRENCODEC_ENABLE_TFLITE "Also link LiteRT/TensorFlow Lite headers/library when provided" OFF)
option(SIRENCODEC_NATIVE_OPT "Enable native CPU tuning flags for Release builds" ON)
option(SIRENCODEC_ARM_OPT "Enable Apple Silicon/AArch64 tuned Release flags" ON)
option(SIRENCODEC_ENABLE_IPO "Enable Release interprocedural optimization when supported" ON)

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

add_library(sirencodec_common STATIC
  src/audio_wav.cpp
  src/metrics.cpp)
target_compile_features(sirencodec_common PUBLIC cxx_std_20)
target_include_directories(sirencodec_common PUBLIC include)

add_library(sirencodec_litert_core STATIC
  src/litert_runner.cpp)
target_compile_features(sirencodec_litert_core PUBLIC cxx_std_20)
target_link_libraries(sirencodec_litert_core PUBLIC sirencodec_common)

find_package(Threads REQUIRED)
target_link_libraries(sirencodec_litert_core PUBLIC Threads::Threads)
target_link_libraries(sirencodec_litert_core PUBLIC ${CMAKE_DL_LIBS})

add_executable(sirencodec_infer src/litert_main.cpp)
target_link_libraries(sirencodec_infer PRIVATE sirencodec_litert_core)

# Compatibility name kept while the project transitions to TFLite-first naming.
add_executable(sirencodec_litert src/litert_main.cpp)
target_link_libraries(sirencodec_litert PRIVATE sirencodec_litert_core)

add_executable(sirencodec_benchmark benchmarks/litert_benchmark.cpp)
target_link_libraries(sirencodec_benchmark PRIVATE sirencodec_litert_core)

add_executable(sirencodec_dataset_benchmark benchmarks/litert_dataset_benchmark.cpp)
target_link_libraries(sirencodec_dataset_benchmark PRIVATE sirencodec_litert_core)

# Compatibility name for older benchmark commands.
add_executable(sirencodec_litert_benchmark benchmarks/litert_benchmark.cpp)
target_link_libraries(sirencodec_litert_benchmark PRIVATE sirencodec_litert_core)

add_executable(sirencodec_tests tests/litert_smoke_tests.cpp)
target_link_libraries(sirencodec_tests PRIVATE sirencodec_litert_core)

set(SIRENCODEC_ALL_TARGETS
  sirencodec_common
  sirencodec_litert_core
  sirencodec_infer
  sirencodec_litert
  sirencodec_benchmark
  sirencodec_dataset_benchmark
  sirencodec_litert_benchmark
  sirencodec_tests)

set(SIRENCODEC_INSTALL_TARGETS
  sirencodec_infer
  sirencodec_litert
  sirencodec_benchmark
  sirencodec_dataset_benchmark
  sirencodec_litert_benchmark
  sirencodec_tests)

function(sirencodec_add_release_compile_option target option)
  string(MAKE_C_IDENTIFIER "SIRENCODEC_HAS_${option}" option_var)
  check_cxx_compiler_flag("${option}" "${option_var}")
  if (${option_var})
    target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:${option}>")
  endif()
endfunction()

find_path(SNDFILE_INCLUDE_DIR sndfile.h)
find_library(SNDFILE_LIBRARY NAMES sndfile)
if (SNDFILE_INCLUDE_DIR AND SNDFILE_LIBRARY)
  target_compile_definitions(sirencodec_common PUBLIC SIRENCODEC_HAVE_SNDFILE=1)
  target_include_directories(sirencodec_common PUBLIC "${SNDFILE_INCLUDE_DIR}")
  target_link_libraries(sirencodec_common PUBLIC "${SNDFILE_LIBRARY}")
  message(STATUS "libsndfile found: dataset FLAC/WAV loading enabled")
else()
  message(STATUS "libsndfile not found: built-in WAV loader only")
endif()

if (SIRENCODEC_ENABLE_TFLITE)
  set(LITERT_INCLUDE_DIR "" CACHE PATH "LiteRT C/C++ SDK include root containing litert/c/litert_common.h")
  set(LITERT_LIBRARY "" CACHE FILEPATH "LiteRT runtime library, usually libLiteRt.so")
  find_path(LITERT_INCLUDE_DIR litert/c/litert_common.h)
  find_library(LITERT_LIBRARY NAMES LiteRt libLiteRt.so)
  if (NOT LITERT_INCLUDE_DIR OR NOT LITERT_LIBRARY)
    message(FATAL_ERROR
      "SIRENCODEC_ENABLE_TFLITE=ON requires LiteRT/TensorFlow Lite headers and libLiteRt.so. "
      "Set LITERT_INCLUDE_DIR/LITERT_LIBRARY or install/build LiteRT first.")
  endif()
  target_compile_definitions(sirencodec_litert_core PUBLIC SIRENCODEC_ENABLE_TFLITE=1)
  target_include_directories(sirencodec_litert_core PUBLIC "${LITERT_INCLUDE_DIR}")
  target_link_libraries(sirencodec_litert_core PUBLIC "${LITERT_LIBRARY}")
endif()

if (MSVC)
  foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
    target_compile_options(${target} PRIVATE /W4)
  endforeach()
  if (SIRENCODEC_WARNINGS_AS_ERRORS)
    foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
      target_compile_options(${target} PRIVATE /WX)
    endforeach()
  endif()
else()
  foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wdouble-promotion)
  endforeach()
  if (SIRENCODEC_WARNINGS_AS_ERRORS)
    foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
      target_compile_options(${target} PRIVATE -Werror)
    endforeach()
  endif()
endif()

if (SIRENCODEC_NATIVE_OPT AND NOT MSVC)
  if (SIRENCODEC_ARM_OPT AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"))
    foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
      sirencodec_add_release_compile_option(${target} -mcpu=native)
      sirencodec_add_release_compile_option(${target} -ffast-math)
      sirencodec_add_release_compile_option(${target} -ffp-contract=fast)
      sirencodec_add_release_compile_option(${target} -fomit-frame-pointer)
    endforeach()
  else()
    foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
      sirencodec_add_release_compile_option(${target} -march=native)
      sirencodec_add_release_compile_option(${target} -mtune=native)
      sirencodec_add_release_compile_option(${target} -ffast-math)
    endforeach()
  endif()
endif()

if (SIRENCODEC_ENABLE_IPO)
  check_ipo_supported(RESULT SIRENCODEC_IPO_SUPPORTED OUTPUT SIRENCODEC_IPO_OUTPUT)
  if (SIRENCODEC_IPO_SUPPORTED)
    foreach(target IN LISTS SIRENCODEC_ALL_TARGETS)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endforeach()
  else()
    message(STATUS "IPO/LTO not enabled: ${SIRENCODEC_IPO_OUTPUT}")
  endif()
endif()

install(TARGETS ${SIRENCODEC_INSTALL_TARGETS} RUNTIME DESTINATION bin)
