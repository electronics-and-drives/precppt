cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(precppt VERSION 0.1.0)

## Make sure libTorch is installed
find_package(Torch REQUIRED)
include_directories(${PROJECT_SOURCE_DIR})

## Shared Library libpca.so
add_library(pcp SHARED pcp.cpp)

## Static Library libpcp.a
#add_library(pcl STATIC pcp.cpp)
#set_property(TARGET pcp PROPERTY POSITION_INDEPENDENT_CODE ON)

include (GenerateExportHeader)          
GENERATE_EXPORT_HEADER( pcp
                        BASE_NAME pcp
                        EXPORT_MACRO_NAME pcp_EXPORTS
                        EXPORT_FILE_NAME pcp_EXPORTS.h
                        STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC )

set_property(TARGET pcp PROPERTY CXX_STANDARD 14)
target_link_libraries(pcp "${TORCH_LIBRARIES}")

## Executable for demo purposes
add_executable(pci pci.cpp)
set_property(TARGET pci PROPERTY CXX_STANDARD 14)
target_link_libraries(pci pcp "${TORCH_LIBRARIES}")


