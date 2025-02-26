cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(mini_llm_inference LANGUAGES C CXX VERSION 0.1.0)

option(ENABLE_TEST "Build with TEST." OFF)

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/core DIR_CORE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kern DIR_KERN)
add_library(mini_llm_inference STATIC ${DIR_SRC} ${DIR_CORE} ${DIR_KERN})
add_library(mini_llm_inference_shared SHARED ${DIR_SRC} ${DIR_CORE} ${DIR_KERN})

target_include_directories(mini_llm_inference PUBLIC include ${PROJECT_SOURCE_DIR}/src)
target_include_directories(mini_llm_inference_shared PUBLIC include ${PROJECT_SOURCE_DIR}/src)

add_subdirectory(application)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()