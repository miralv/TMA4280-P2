project(TMA4280_project1)
cmake_minimum_required( VERSION 2.8)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)



foreach (prog poisson_parallel)
    add_executable(${prog} ${prog}.cpp)
    target_include_directories(${prog} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_options(${prog} PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
    target_compile_definitions(${prog} PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
    target_link_libraries(${prog} ${MPI_CXX_LIBRARIES} )
    set_target_properties(${prog} PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endforeach()

#The files using only openmp
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(poisson_parallel poisson_parallel.cpp)


