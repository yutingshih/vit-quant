cmake_minimum_required(VERSION 3.18...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
file(GLOB KERNEL_CUDA_SOURCES CONFIGURE_DEPENDS
    src/kernels/*.cu
)
nanobind_add_module(kernels NB_STATIC
    src/kernels/binding.cu
    ${KERNEL_CUDA_SOURCES}
)
target_link_libraries(kernels PRIVATE CUDA::cublas)
install(TARGETS kernels DESTINATION ${SKBUILD_PROJECT_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()
