cmake_minimum_required(VERSION 3.20)
project(hpc_cavity LANGUAGES CXX)

add_library(cavity_core src/cavity.cpp)
target_include_directories(cavity_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(cavity_core PUBLIC cxx_std_17)

add_executable(cavity_serial src/main.cpp)
target_link_libraries(cavity_serial PRIVATE cavity_core)

add_executable(poisson_serial_bench src/poisson_bench.cpp)
target_link_libraries(poisson_serial_bench PRIVATE cavity_core)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_library(cavity_core_openmp src/cavity.cpp)
  target_include_directories(cavity_core_openmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_features(cavity_core_openmp PUBLIC cxx_std_17)
  target_compile_definitions(cavity_core_openmp PRIVATE CAVITY_USE_OPENMP)
  target_link_libraries(cavity_core_openmp PUBLIC OpenMP::OpenMP_CXX)

  add_executable(cavity_openmp src/main.cpp)
  target_compile_definitions(cavity_openmp PRIVATE CAVITY_USE_OPENMP)
  target_link_libraries(cavity_openmp PRIVATE cavity_core_openmp)
endif()

if(WIN32)
  # MS-MPI installs the compiler headers separately from mpiexec.
  find_path(MSMPI_INCLUDE_DIR mpi.h
    PATHS "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
  find_library(MSMPI_LIBRARY msmpi
    PATHS "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
  if(MSMPI_INCLUDE_DIR AND MSMPI_LIBRARY)
    add_library(cavity_mpi_dependency INTERFACE)
    target_include_directories(cavity_mpi_dependency INTERFACE ${MSMPI_INCLUDE_DIR})
    target_link_libraries(cavity_mpi_dependency INTERFACE ${MSMPI_LIBRARY})
    set(CAVITY_MPI_FOUND TRUE)
  endif()
else()
  find_package(MPI COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    add_library(cavity_mpi_dependency INTERFACE)
    target_link_libraries(cavity_mpi_dependency INTERFACE MPI::MPI_CXX)
    set(CAVITY_MPI_FOUND TRUE)
  endif()
endif()

if(CAVITY_MPI_FOUND)
  add_executable(cavity_mpi src/mpi_main.cpp src/mpi_solver.cpp)
  target_include_directories(cavity_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_features(cavity_mpi PRIVATE cxx_std_17)
  target_link_libraries(cavity_mpi PRIVATE cavity_mpi_dependency)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(cavity_tests tests/test_solver.cpp)
  target_link_libraries(cavity_tests PRIVATE cavity_core)
  add_test(NAME cavity_serial_checks COMMAND cavity_tests)
  if(OpenMP_CXX_FOUND)
    add_executable(cavity_openmp_tests tests/test_solver.cpp)
    target_link_libraries(cavity_openmp_tests PRIVATE cavity_core_openmp)
    add_test(NAME cavity_openmp_checks COMMAND cavity_openmp_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME serial_openmp_equivalence
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_openmp.py
          $<TARGET_FILE:cavity_serial> $<TARGET_FILE:cavity_openmp> ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  endif()
  if(CAVITY_MPI_FOUND)
    find_package(Python3 COMPONENTS Interpreter)
    if(WIN32)
      find_program(CAVITY_MPIEXEC mpiexec
        PATHS "C:/Program Files/Microsoft MPI/Bin")
    else()
      set(CAVITY_MPIEXEC ${MPIEXEC_EXECUTABLE})
    endif()
    if(Python3_Interpreter_FOUND AND CAVITY_MPIEXEC)
      add_test(NAME serial_mpi_equivalence
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_mpi.py
          ${CAVITY_MPIEXEC} $<TARGET_FILE:cavity_serial> $<TARGET_FILE:cavity_mpi>
          ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  endif()
endif()
