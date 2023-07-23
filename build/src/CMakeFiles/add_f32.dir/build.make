# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/protox/habana/HPU_Benchmarking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/protox/habana/HPU_Benchmarking/build

# Include any dependencies generated for this target.
include src/CMakeFiles/add_f32.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/add_f32.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/add_f32.dir/flags.make

src/add_f32_x86.o: ../kernel/add_f32.c
src/add_f32_x86.o: /usr/bin/tpc-clang
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/protox/habana/HPU_Benchmarking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating add_f32_x86.o"
	cd /home/protox/habana/HPU_Benchmarking/build/src && cp /home/protox/habana/HPU_Benchmarking/./kernel/add_f32.c /home/protox/habana/HPU_Benchmarking/build/src
	cd /home/protox/habana/HPU_Benchmarking/build/src && /usr/bin/tpc-clang -Wall -Werror -march=gaudi -O2 -I/usr/lib/habanatools/include/ -I/home/protox/habana/HPU_Benchmarking/kernels/include ./add_f32.c -c
	cd /home/protox/habana/HPU_Benchmarking/build/src && objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./add_f32.o add_f32_x86.o
	cd /home/protox/habana/HPU_Benchmarking/build/src && /usr/bin/tpc-llvm-objdump --triple tpc -d -j .text --no-show-raw-insn --no-leading-addr --mcpu=gaudi .//add_f32.o > /home/protox/habana/HPU_Benchmarking/build/src//add_f32.tpcasm

# Object files for target add_f32
add_f32_OBJECTS =

# External object files for target add_f32
add_f32_EXTERNAL_OBJECTS = \
"/home/protox/habana/HPU_Benchmarking/build/src/add_f32_x86.o"

src/libadd_f32.a: src/add_f32_x86.o
src/libadd_f32.a: src/CMakeFiles/add_f32.dir/build.make
src/libadd_f32.a: src/CMakeFiles/add_f32.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/protox/habana/HPU_Benchmarking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libadd_f32.a"
	cd /home/protox/habana/HPU_Benchmarking/build/src && $(CMAKE_COMMAND) -P CMakeFiles/add_f32.dir/cmake_clean_target.cmake
	cd /home/protox/habana/HPU_Benchmarking/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add_f32.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/add_f32.dir/build: src/libadd_f32.a

.PHONY : src/CMakeFiles/add_f32.dir/build

src/CMakeFiles/add_f32.dir/clean:
	cd /home/protox/habana/HPU_Benchmarking/build/src && $(CMAKE_COMMAND) -P CMakeFiles/add_f32.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/add_f32.dir/clean

src/CMakeFiles/add_f32.dir/depend: src/add_f32_x86.o
	cd /home/protox/habana/HPU_Benchmarking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/protox/habana/HPU_Benchmarking /home/protox/habana/HPU_Benchmarking/src /home/protox/habana/HPU_Benchmarking/build /home/protox/habana/HPU_Benchmarking/build/src /home/protox/habana/HPU_Benchmarking/build/src/CMakeFiles/add_f32.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/add_f32.dir/depend

