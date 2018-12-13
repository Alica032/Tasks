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
CMAKE_COMMAND = /home/alica032/Documents/csc/gpu/clion-2018.2.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/alica032/Documents/csc/gpu/clion-2018.2.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alica032/Documents/csc/gpu/Tasks

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mandelbrot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mandelbrot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mandelbrot.dir/flags.make

../src/cl/mandelbrot_cl.h: ../src/cl/mandelbrot.cl
../src/cl/mandelbrot_cl.h: libs/gpu/hexdumparray
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../src/cl/mandelbrot_cl.h"
	libs/gpu/hexdumparray /home/alica032/Documents/csc/gpu/Tasks/src/cl/mandelbrot.cl /home/alica032/Documents/csc/gpu/Tasks/src/cl/mandelbrot_cl.h mandelbrot_kernel

CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o: CMakeFiles/mandelbrot.dir/flags.make
CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o: ../src/main_mandelbrot.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o -c /home/alica032/Documents/csc/gpu/Tasks/src/main_mandelbrot.cpp

CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alica032/Documents/csc/gpu/Tasks/src/main_mandelbrot.cpp > CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.i

CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alica032/Documents/csc/gpu/Tasks/src/main_mandelbrot.cpp -o CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.s

# Object files for target mandelbrot
mandelbrot_OBJECTS = \
"CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o"

# External object files for target mandelbrot
mandelbrot_EXTERNAL_OBJECTS =

mandelbrot: CMakeFiles/mandelbrot.dir/src/main_mandelbrot.cpp.o
mandelbrot: CMakeFiles/mandelbrot.dir/build.make
mandelbrot: libs/clew/liblibclew.a
mandelbrot: libs/gpu/liblibgpu.a
mandelbrot: libs/images/liblibimages.a
mandelbrot: libs/utils/liblibutils.a
mandelbrot: libs/gpu/liblibgpu.a
mandelbrot: libs/utils/liblibutils.a
mandelbrot: libs/clew/liblibclew.a
mandelbrot: /usr/lib/x86_64-linux-gnu/libSM.so
mandelbrot: /usr/lib/x86_64-linux-gnu/libICE.so
mandelbrot: /usr/lib/x86_64-linux-gnu/libX11.so
mandelbrot: /usr/lib/x86_64-linux-gnu/libXext.so
mandelbrot: CMakeFiles/mandelbrot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable mandelbrot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mandelbrot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mandelbrot.dir/build: mandelbrot

.PHONY : CMakeFiles/mandelbrot.dir/build

CMakeFiles/mandelbrot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mandelbrot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mandelbrot.dir/clean

CMakeFiles/mandelbrot.dir/depend: ../src/cl/mandelbrot_cl.h
	cd /home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alica032/Documents/csc/gpu/Tasks /home/alica032/Documents/csc/gpu/Tasks /home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug /home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug /home/alica032/Documents/csc/gpu/Tasks/cmake-build-debug/CMakeFiles/mandelbrot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mandelbrot.dir/depend
