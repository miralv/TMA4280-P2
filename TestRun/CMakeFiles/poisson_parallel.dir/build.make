# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mira/Documents/NTNU/TMA4280-P2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mira/Documents/NTNU/TMA4280-P2/TestRun

# Include any dependencies generated for this target.
include CMakeFiles/poisson_parallel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/poisson_parallel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/poisson_parallel.dir/flags.make

CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o: CMakeFiles/poisson_parallel.dir/flags.make
CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o: ../poisson_parallel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mira/Documents/NTNU/TMA4280-P2/TestRun/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o -c /home/mira/Documents/NTNU/TMA4280-P2/poisson_parallel.cpp

CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mira/Documents/NTNU/TMA4280-P2/poisson_parallel.cpp > CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.i

CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mira/Documents/NTNU/TMA4280-P2/poisson_parallel.cpp -o CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.s

# Object files for target poisson_parallel
poisson_parallel_OBJECTS = \
"CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o"

# External object files for target poisson_parallel
poisson_parallel_EXTERNAL_OBJECTS =

poisson_parallel: CMakeFiles/poisson_parallel.dir/poisson_parallel.cpp.o
poisson_parallel: CMakeFiles/poisson_parallel.dir/build.make
poisson_parallel: libcommon.a
poisson_parallel: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
poisson_parallel: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
poisson_parallel: CMakeFiles/poisson_parallel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mira/Documents/NTNU/TMA4280-P2/TestRun/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable poisson_parallel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/poisson_parallel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/poisson_parallel.dir/build: poisson_parallel

.PHONY : CMakeFiles/poisson_parallel.dir/build

CMakeFiles/poisson_parallel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/poisson_parallel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/poisson_parallel.dir/clean

CMakeFiles/poisson_parallel.dir/depend:
	cd /home/mira/Documents/NTNU/TMA4280-P2/TestRun && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mira/Documents/NTNU/TMA4280-P2 /home/mira/Documents/NTNU/TMA4280-P2 /home/mira/Documents/NTNU/TMA4280-P2/TestRun /home/mira/Documents/NTNU/TMA4280-P2/TestRun /home/mira/Documents/NTNU/TMA4280-P2/TestRun/CMakeFiles/poisson_parallel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/poisson_parallel.dir/depend

