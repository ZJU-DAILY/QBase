cmake_minimum_required(VERSION 3.5)
project(m3v VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PROJECT_ROOT_PATH "${CMAKE_SOURCE_DIR}")
message(STATUS "root_path: ${CMAKE_SOURCE_DIR}")
add_definitions(-DPROJECT_ROOT_PATH="${PROJECT_ROOT_PATH}")
set(PostgreSQL_ADDITIONAL_VERSIONS "16")
list(APPEND CMAKE_PREFIX_PATH "/home/tanboyu/cpp_workspace/qbase_data")
include_directories("/home/tanboyu/cpp_workspace/qbase_data/include/postgresql/server")
find_package(PostgreSQL REQUIRED COMPONENTS Server)
if(PostgreSQL_FOUND)
    message(STATUS "PostgreSQL_INCLUDE_DIRS: ${PostgreSQL_INCLUDE_DIRS}")
endif()

# Add Control and SQL Files
set(EXT_CONTROL_FILE ${PROJECT_NAME}.control)
set(EXT_SQL_FILE ${PROJECT_NAME}--1.0.sql)

## build rocksdb
add_custom_target(build_rocksdb
    COMMAND ${CMAKE_MAKE_PROGRAM} -C /home/tanboyu/cpp_workspace/third_party/rocksdb
    DEPENDS /home/tanboyu/cpp_workspace/third_party/rocksdb/Makefile
    COMMENT "Building RocksDB"
)

# Add Compile Options
add_definitions(-DPLATFORM_UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -maes -mavx -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -pthread -march=native")
set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g2 -ggdb -pthread -Wall -DINFO")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -mmwaitx ${CMAKE_EXE_LINKER_FLAGS} -fPIC")
message(STATUS "linker: ${CMAKE_EXE_LINKER_FLAGS}")

message(STATUS "vector path: ${CMAKE_SOURCE_DIR}/src/vector.h")

# Add Source Files
add_library(${PROJECT_NAME} SHARED
    src/a3v/spi_filter.cpp
    src/a3v/simd_func.cpp
    src/a3v/memory_a3v.cpp
    src/a3v/disk_a3v.cpp
    src/a3v/init.cpp
    src/a3v/m3v.cpp 
    src/a3v/hnsw.cpp 
    src/a3v/m3vbuild.cpp 
    src/a3v/m3vinsert.cpp 
    src/a3v/m3vscan.cpp 
    src/a3v/m3vutils.cpp 
    src/a3v/m3vvaccum.cpp
    src/a3v/vector.cpp
    src/a3v/util.cpp
    src/a3v/a3v_async_server.cpp
    src/record_cache/record_io.cpp
    src/record_cache/lru_index_pointer.cpp
    src/record_cache/validity_mask.cpp
    src/AuxiliaryIndex/page_sort_index.cpp
)

file(GLOB ROCKSDB_DYLIBS "/home/tanboyu/cpp_workspace/third_party/rocksdb/*.so.*")
message(STATUS "ROCKSDB_DYLIBS:${ROCKSDB_DYLIBS}")
message(STATUS "CMAKE_SOURCE_DIR:${CMAKE_SOURCE_DIR}")

target_link_libraries(${PROJECT_NAME} ${ROCKSDB_DYLIBS})
# set_target_properties(${PROJECT_NAME} PROPERTIES
#                       BUILD_WITH_INSTALL_RPATH TRUE
#                       INSTALL_RPATH ${ROCKSDB_DYLIBS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
# target_compile_options(${PROJECT_NAME} PRIVATE -Wall -mavx2)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC /data/include/postgresql/server)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/record_cache/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/ /home/tanboyu/cpp_workspace/OneDB186/contrib/m3v/third_party/rocksdb/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/contrib/m3v/src/utils)
target_include_directories(${PROJECT_NAME} PUBLIC /home/tanboyu/cpp_workspace/OneDB186/contrib/m3v/third_party/hnsw/hnswlib)
# target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/contrib/m3v/src/a3v)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/AuxiliaryIndex/)
if(WIN32)
    message(STATUS "WIN32")
    target_include_directories(${PROJECT_NAME} PUBLIC ${PostgreSQL_TYPE_INCLUDE_DIR}/port/win32)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${PostgreSQL_LIBRARY_DIRS}/postgres.lib)
    if(MSVC)
        target_include_directories(${PROJECT_NAME} PUBLIC ${PostgreSQL_TYPE_INCLUDE_DIR}/port/win32_msvc)
        target_compile_options(${PROJECT_NAME} PRIVATE /TC)
    endif(MSVC)
endif(WIN32)

set(PostgreSQL_PG_CONFIG  $ENV{PGHOME}/bin/pg_config)
if (NOT PostgreSQL_PG_CONFIG)
    message(FATAL_ERROR "Unable to find 'pg_config'")
endif ()
message(STATUS "Using pg_config ${PostgreSQL_PG_CONFIG}")

## $> pg_config --sharedir
## /data/share/postgresql
execute_process(
        COMMAND ${PostgreSQL_PG_CONFIG} --sharedir
        OUTPUT_VARIABLE PostgreSQL_SHAREDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
        COMMAND ${PostgreSQL_PG_CONFIG} --pkglibdir
        OUTPUT_VARIABLE PostgreSQL_PKGLIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "EXTENSION_FILE: ${CMAKE_CURRENT_SOURCE_DIR}/${EXT_CONTROL_FILE}")
message(STATUS "SQL_FILE: ${CMAKE_CURRENT_SOURCE_DIR}/${EXT_SQL_FILE}")
message(STATUS "DESTINATION: ${PostgreSQL_SHAREDIR}/extension")

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/${EXT_CONTROL_FILE}
    DESTINATION "${PostgreSQL_SHAREDIR}/extension")

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/${EXT_SQL_FILE}
    DESTINATION "${PostgreSQL_SHAREDIR}/extension")

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION ${PostgreSQL_PKGLIBDIR})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd+crypto+crc")
# set(CMAKE_C_COMPILER arm-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER arm-linux-gnu-g++)
add_compile_options(-O3)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_MAVX)
if(COMPILER_SUPPORTS_MAVX)
    message("Compiler supports AVX. USE_AVX will be defined.")
    add_compile_options(-mavx2)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVX)
endif()

message(STATUS "PROJECT_VERSION: ${PROJECT_VERSION}")
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "EXT_CONTROL_FILE: ${EXT_CONTROL_FILE}")
message(STATUS "EXT_SQL_FILE: ${EXT_SQL_FILE}")
message (STATUS "UNIX: CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message (STATUS "UNIX: CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "PostgreSQL_SHAREDIR: ${PostgreSQL_SHAREDIR}")
message(STATUS "PostgreSQL_PKGLIBDIR: ${PostgreSQL_PKGLIBDIR}")

# add_subdirectory(third_party/lrucache11)

# add_subdirectory(third_party/googletest)
# enable_testing()
# file(GLOB TEST_SOURCES "src/tests/*.cpp")
# file(GLOB RECORD_SOURCES "src/record_cache/*.cpp")
# file(GLOB PAGE_SORT_SOURCES "src/AuxiliaryIndex/*.cpp")
# set(TEST_FILES ${TEST_SOURCES} ${RECORD_SOURCES} ${PAGE_SORT_SOURCES})
# message(STATUS "TEST_FILES:${TEST_FILES}")
# message(STATUS "RECORD_SOURCES:${RECORD_SOURCES}")
# add_executable(TEST_M3V ${TEST_FILES})
# target_link_libraries(TEST_M3V gtest gtest_main ${ROCKSDB_DYLIBS} LRUCache11)
# target_include_directories(TEST_M3V PRIVATE /home/tanboyu/cpp_workspace/third_party/lrucache11)
# message("source directory: /home/tanboyu/cpp_workspace/third_party/rocksdb/include")
# target_include_directories(TEST_M3V PRIVATE /home/tanboyu/cpp_workspace/third_party/rocksdb/include ${CMAKE_SOURCE_DIR}/src/record_cache/include)
# target_include_directories(TEST_M3V PRIVATE /data/include/postgresql/server ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/utils)
# target_include_directories(TEST_M3V PRIVATE ${CMAKE_SOURCE_DIR}/src/AuxiliaryIndex/)
# target_include_directories(TEST_M3V PRIVATE /home/tanboyu/cpp_workspace/third_party/hnsw/hnswlib/)
# add_test(NAME TEST_M3V_ COMMAND TEST_M3V)

# add_executable(VectorGenerate ${CMAKE_SOURCE_DIR}/src/tests/generate_vector.cpp)
# cmake -DCMAKE_INSTALL_PREFIX=/data/data -DLIBRARYONLY=ON -DSEEK_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
# make -j$(nproc)
# make install
# linux: export LD_LIBRARY_PATH=/home/jack/cpp_workspace/wrapdir/OneDb2/contrib/m3v/third_party/rocksdb/:$DYLD_LIBRARY_PATH
# macos: export DYLD_LIBRARY_PATH=/home/jack/cpp_workspace/wrapdir/OneDb2/contrib/m3v/third_party/rocksdb/:$DYLD_LIBRARY_PATH
## restart to load rocksdb so files.
# pg_ctl -D /data/data -l logfile stop
# pg_ctl -D /data/data -l logfile start
