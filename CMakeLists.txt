project(TMA4280-P2)
cmake_minimum_required( VERSION 2.8)

enable_language(C)
enable_language(Fortran)
enable_language(CXX)


find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(poisson_parallel poisson_parallel.cpp)
target_include_directories(poisson_parallel PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(poisson_parallel PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(poisson_parallel PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
set_target_properties(poisson_parallel PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})


# OpenMP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Add a library with common sources
add_library(common fst.f)

# Link to libraries
target_link_libraries(poisson_parallel common ${MPI_CXX_LIBRARIES})

# We want to link this as a C++ program
# set_target_properties(poisson_parallel PROPERTIES LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})