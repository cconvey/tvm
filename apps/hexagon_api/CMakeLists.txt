cmake_minimum_required(VERSION 3.2)

project(hexagon_api)

include(ExternalProject)

# Required variables:
#   ANDROID_ABI
#   ANDROID_PLATFORM
#   USE_ANDROID_TOOLCHAIN (Android toolchain .cmake file)
#   USE_HEXAGON_ARCH
#   USE_HEXAGON_SDK
#   USE_HEXAGON_TOOLCHAIN (Path to Hexagon toolchain ending with "Tools")
# Optional variable:
#   USE_OUTPUT_BINARY_DIR (Path to copy the output binaries to)

if(USE_CCACHE) # True for AUTO, ON, /path/to/ccache
  if("${USE_CCACHE}" STREQUAL "AUTO") # Auto mode
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      message(STATUS "Found the path to ccache, enabling ccache")
      set(PATH_TO_CCACHE ccache)
    else()
      message(STATUS "Didn't find the path to CCACHE, disabling ccache")
    endif(CCACHE_FOUND)
    #elseif("${USE_CCACHE}" MATCHES ${IS_TRUE_PATTERN})
  elseif(USE_CCACHE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      message(STATUS "Found the path to ccache, enabling ccache")
      set(PATH_TO_CCACHE ccache)
    else()
      message(FATAL_ERROR "Cannot find ccache. Set USE_CCACHE mode to AUTO or OFF to build without ccache. USE_CCACHE=" "${USE_CCACHE}")
    endif(CCACHE_FOUND)
  else() # /path/to/ccache
    set(PATH_TO_CCACHE USE_CCACHE)
    message(STATUS "Setting ccache path to " "${PATH_TO_CCACHE}")
  endif()
  # Set the flag for ccache
  set(CXX_COMPILER_LAUNCHER "${PATH_TO_CCACHE}")
endif(USE_CCACHE)

message(WARNING "CXX_COMPILER_LAUNCHER = '${CXX_COMPILER_LAUNCHER}'")

set(TVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../..")

if(DEFINED USE_OUTPUT_BINARY_DIR)
  set(HEXAGON_API_BINARY_DIR "${USE_OUTPUT_BINARY_DIR}")
else()
  set(HEXAGON_API_BINARY_DIR "${CMAKE_BINARY_DIR}/hexagon_rpc")
endif()
file(MAKE_DIRECTORY ${HEXAGON_API_BINARY_DIR})

# Build X86 binaries:
# - tvm_rpc_x86

#message(FATAL_ERROR "MAKE='${MAKE}' CMAKE_MAKE_PROGRAM='${CMAKE_MAKE_PROGRAM}'")

ExternalProject_Add(x86_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  #BUILD_COMMAND $(MAKE) runtime tvm_rpc
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} runtime tvm_rpc
  CMAKE_ARGS
    "-DUSE_CCACHE=${USE_CCACHE}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CXX_COMPILER_LAUNCHER}"
    "-DUSE_HEXAGON_TOOLCHAIN=${USE_HEXAGON_TOOLCHAIN}"
    "-DCMAKE_CXX_STANDARD=14"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DBUILD_STATIC_RUNTIME=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(x86_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(x86_tvm_runtime_rpc copy_rpc_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/tvm_rpc
    ${HEXAGON_API_BINARY_DIR}/tvm_rpc_x86
  DEPENDEES install
)

# Build Android binaries:
# - libtvm_runtime.so
# - tvm_rpc_android

ExternalProject_Add(android_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} runtime tvm_rpc
  #BUILD_COMMAND $(MAKE) runtime tvm_rpc
  CMAKE_ARGS
    "-DUSE_CCACHE=${USE_CCACHE}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CXX_COMPILER_LAUNCHER}"
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DCMAKE_CXX_STANDARD=14"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DUSE_ALTERNATIVE_LINKER=OFF"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(android_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(android_tvm_runtime_rpc copy_runtime
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.so
    ${HEXAGON_API_BINARY_DIR}
  DEPENDEES install
)
ExternalProject_Add_Step(android_tvm_runtime_rpc copy_rpc_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/tvm_rpc
    ${HEXAGON_API_BINARY_DIR}/tvm_rpc_android
  DEPENDEES install
)


# Build Hexagon binaries:
# - libhexagon_rpc_skel.so
# - libtvm_runtime.a

ExternalProject_Add(hexagon_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} runtime hexagon_rpc_sim
  #BUILD_COMMAND $(MAKE) runtime hexagon_rpc_sim
  CMAKE_ARGS
    "-DUSE_CCACHE=${USE_CCACHE}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CXX_COMPILER_LAUNCHER}"
    "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=OFF"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DBUILD_STATIC_RUNTIME=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DUSE_ALTERNATIVE_LINKER=OFF"
    "-DUSE_CUSTOM_LOGGING=ON"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(hexagon_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(hexagon_tvm_runtime_rpc copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.a
    ${BINARY_DIR}/libhexagon_rpc_skel.so
    ${BINARY_DIR}/libhexagon_rpc_sim.so
    ${HEXAGON_API_BINARY_DIR}
  DEPENDEES install
)

configure_file("${TVM_SOURCE_DIR}/src/runtime/hexagon/rpc/android_bash.sh.template"
  ${HEXAGON_API_BINARY_DIR} COPYONLY)
