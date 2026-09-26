cmake_minimum_required(VERSION 3.18)
project(w1ax_operator_replay LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(../../third_party/llama.cpp/ggml ggml-build)
add_executable(w1ax_operator_replay replay.cpp)
target_link_libraries(w1ax_operator_replay PRIVATE ggml)
