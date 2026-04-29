cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)
option(S2_BUILD_SHARED_LIBRARIES   "Compile S2 as a dynamic link library(dll) and a shared(.lib) library as well."   OFF)

# ---------------------------------------------------------------------------
# GGML (git submodule — not modified)
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # Enable ggml's CUDA Graphs path. Profiling on A100 showed the
    # autoregressive decode loop (eval_cached + fast_decode×N codebooks
    # per step) was bottlenecked on per-token CUDA launch / sync
    # overhead — sm utilisation hovered near zero between bursts.
    # ggml's automatic capture-replay (warmup-then-replay) eliminates
    # the per-call CPU code path on subsequent identical-shape graph
    # computes, which is exactly the fast_decode loop's pattern.
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
if(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# s2 executable
# ---------------------------------------------------------------------------
set(S2_SOURCES
    src/s2_export_api.cpp
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/s2_server.cpp
    src/main.cpp
)

if(S2_BUILD_SHARED_LIBRARIES)
add_executable(s2 ${S2_SOURCES})
add_library(s2_dll SHARED ${S2_SOURCES})
add_library(s2_lib STATIC ${S2_SOURCES})

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src          # for ggml-common.h etc.
)
target_include_directories(s2_dll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src          # for ggml-common.h etc.
)
target_include_directories(s2_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src          # for ggml-common.h etc.
)

target_link_libraries(s2 PRIVATE ggml)
target_link_libraries(s2_dll PRIVATE ggml)
target_link_libraries(s2_lib PRIVATE ggml)


target_compile_definitions(s2_dll PRIVATE S2_LIBRARY)
target_compile_definitions(s2_lib PRIVATE S2_LIBRARY)
if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
	target_compile_definitions(s2_dll PRIVATE GGML_USE_VULKAN)
	target_compile_definitions(s2_lib PRIVATE GGML_USE_VULKAN)
endif()
if(S2_CUDA)
    target_compile_definitions(s2 PRIVATE GGML_USE_CUDA)
	target_compile_definitions(s2_dll PRIVATE GGML_USE_CUDA)
	target_compile_definitions(s2_lib PRIVATE GGML_USE_CUDA)
endif()


# Platform-specific
if(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
	target_link_libraries(s2_dll PRIVATE pthread m)
	target_link_libraries(s2_lib PRIVATE pthread m)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)
install(TARGETS s2_dll RUNTIME DESTINATION bin)
install(TARGETS s2_lib RUNTIME DESTINATION bin)

else()
add_executable(s2 ${S2_SOURCES})

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src          # for ggml-common.h etc.
)

target_link_libraries(s2 PRIVATE ggml)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
endif()
if(S2_CUDA)
    target_compile_definitions(s2 PRIVATE GGML_USE_CUDA)
endif()
if(S2_METAL)
    target_compile_definitions(s2 PRIVATE GGML_USE_METAL)
endif()

# Platform-specific
if(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)

endif()









