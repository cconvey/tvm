#!/bin/bash
set -o errexit
set -o nounset
set -o errtrace

#-------------------------------------------------------------------------------
# ABOUT THIS SCRIPT
#-------------------------------------------------------------------------------
#
# This is a convenience script to orchestrate the configuration and building of
# TVM for running our Hello, World! walkthrough on Octogon + Hexagon.
#
# PRECONDITIONS:
#
# - The current working directory is the root of the TVM source tree.
#
# - The 'build' subdirectories (see the variables TVM_BUILD_DIR and
#   TVM_HEXAGON_API_BUILD_DIR, below) either doesn't exist, or have content
#   that won't cause problems when this script runs 'cmake' and 'make'
#   operations on them.
#
# - The Python environment / system environment in which this script runs
#   meets the various prerequisits for building TVM, using the Hexagon SDK, etc.
#
#
# POSTCONDITIONS:
#
# - TVM and the TVM Hexagon API software has been built (but not installed).
#
# - A file named 'source-me.sh' has been written into the current working
#   directory. After sourcing that file into a Bash-compatible shell, several
#   shell-aliases are defined for setting up / executing the unit tests.
#   Supporting environment variables are also defined/exported as needed.
#
#-------------------------------------------------------------------------------
# Details the user may wish to override...
#-------------------------------------------------------------------------------

TVM_BUILD_CONFIG="${TVM_BUILD_CONFIG:-$(hostname)}"

OCTOGON_MAKE_PARALLELISM=32

case "${TVM_BUILD_CONFIG}" in
    octogon )
        CMAKE_PROG=/opt/cmake-3.22.0-rc1-linux-x86_64/bin/cmake

        HEXAGON_SDK_VER=4.5.0.3
        HEXAGON_TOOLS_VER=8.5.08
        HEXAGON_SDK_DIR="/opt/qualcomm/hexagon/SDK/${HEXAGON_SDK_VER}"
        ANDROID_TOOLCHAIN_DIR="/opt/Android/android-ndk-r19c"

        # NOTE: These variables intentionally refer to different installation roots for LLVM.
        # The 14.0.0.2 installation is missing at least one library needed by its copy of
        # llvm-config.
        LLVM_BIN_DIR=/opt/qualcomm/hexagon/llvm-14.0.0_2/bin
        LLVM_LIB_DIR=/opt/qualcomm/hexagon/llvm-14.0.0_1/lib

        CLANG_CXX_PROG=/home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
        CLANG_C_PROG=/home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang

        CLANG_CXX_RUNTIME_LIB_DIR=home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/lib

        #BUILD_SYSTEM="Ninja"
        #BUILD_CMD="ninja"
        BUILD_SYSTEM="Unix Makefiles"
        BUILD_CMD="make -j${OCTOGON_MAKE_PARALLELISM}"

        # On Octogon, python3.7's pip module has trouble installing all of packages
        # needed for this script. So we'll assume the user has properly set up a
        # Python 3.6 environment.
        PYTHON3_CMD="python3.6"
        ;;
    octogon-clang15 )

        CLANG_CXX_PROG="${LLVM_BIN_DIR}/clang++"
        CLANG_C_PROG="${LLVM_BIN_DIR}/clang"
        CLANG_CXX_RUNTIME_LIB_DIR="${LLVM_LIB_DIR}/x86_64-unknown-linux-gnu"

        #BUILD_SYSTEM="Ninja"
        #BUILD_CMD="ninja"
        BUILD_SYSTEM="Unix Makefiles"
        BUILD_CMD="make -j${OCTOGON_MAKE_PARALLELISM}"

        # On Octogon, python3.7's pip module has trouble installing all of packages
        # needed for this script. So we'll assume the user has properly set up a
        # Python 3.6 environment.
        PYTHON3_CMD="python3.6"
        ;;
    ainz-docker-ci )
        CMAKE_PROG=/usr/local/bin/cmake

        HEXAGON_SDK_VER=4.5.0.3
        HEXAGON_TOOLS_VER=8.5.08
        HEXAGON_SDK_DIR="/opt/qualcomm/hexagon_sdk"
        ANDROID_TOOLCHAIN_DIR="${HEXAGON_SDK_DIR}/tools/android-ndk-r19c"

        # NOTE: These variables intentionally refer to different installation roots for LLVM.
        # The 14.0.0.2 installation is missing at least one library needed by its copy of
        # llvm-config.
        # LLVM_BIN_DIR=/opt/qualcomm/hexagon/llvm-14.0.0_2/bin
        # LLVM_LIB_DIR=/opt/qualcomm/hexagon/llvm-14.0.0_1/lib

        # CLANG_CXX_PROG=/home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
        # CLANG_C_PROG=/home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/bin/clang

        # CLANG_CXX_RUNTIME_LIB_DIR=home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04/lib

        LLVM_RWDI_ROOT=/mnt/data/r/cconvey/tvm/call-extern-ptr/llvm/llvmorg-15.0.6-RelWithDebInfo-install
        LLVM_DEBUG_ROOT=/mnt/data/r/cconvey/tvm/call-extern-ptr/llvm/llvmorg-15.0.6-Debug-install

        LLVM_ROOT="${LLVM_DEBUG_ROOT}"
        LLVM_BIN_DIR="${LLVM_ROOT}/bin"
        LLVM_LIB_DIR="${LLVM_ROOT}/lib"
        CXX_STDLIB_DIR="${LLVM_ROOT}/lib/x86_64-unknown-linux-gnu"

        CLANG_CXX_PROG="${LLVM_RWDI_ROOT}/bin/clang++"
        CLANG_C_PROG="${LLVM_RWDI_ROOT}/bin/clang"
        CLANG_CXX_RUNTIME_LIB_DIR="${LLVM_LIB_DIR}"


        #BUILD_SYSTEM="Ninja"
        #BUILD_CMD="ninja -v"
        BUILD_SYSTEM="Unix Makefiles"
        BUILD_CMD="make -j${AINZ_MAKE_PARALLELISM:-8}"

        # On Octogon, python3.7's pip module has trouble installing all of packages
        # needed for this script. So we'll assume the user has properly set up a
        # Python 3.6 environment.
        PYTHON3_CMD="python3.6"
        ;;
    ainz-native )
        CMAKE_PROG=/usr/bin/cmake

        HEXAGON_SDK_VER=5.1.0.0
        HEXAGON_TOOLS_VER=8.5.13
        HEXAGON_SDK_DIR="/opt/qualcomm/hexagon/SDK/${HEXAGON_SDK_VER}"
        ANDROID_TOOLCHAIN_DIR=/usr/lib/android-sdk/ndk-bundle

        LLVM_BIN_DIR=/opt/llvm/15.0.3/bin
        LLVM_LIB_DIR=/opt/llvm/15.0.3/lib

        CLANG_CXX_PROG="${LLVM_BIN_DIR}/clang++"
        CLANG_C_PROG="${LLVM_BIN_DIR}/clang"
        CLANG_CXX_RUNTIME_LIB_DIR="${LLVM_LIB_DIR}/x86_64-unknown-linux-gnu"

        #BUILD_SYSTEM="Ninja"
        #BUILD_CMD="ninja"
        BUILD_SYSTEM="Unix Makefiles"
        BUILD_CMD="make -j8"

        PYTHON3_CMD="python3.10"
        ;;
    * )
        echo "NEED TO DEFINE A FEW VARIABLES FOR THIS HOST ('$(hostname)')" >&2
        exit 1
        ;;
esac

HEXAGON_TOOLCHAIN_TOOLS_DIR="${HEXAGON_SDK_DIR}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VER}/Tools"

TVM_HOME="${TVM_HOME:-$(pwd)}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-${TVM_HOME}/build}"

#HOST_CXX_COMPILER=/opt/llvm-github/llvm-project/llvmorg-14.0.3-install/bin/clang++
#HOST_C_COMPILER=/opt/llvm-github/llvm-project/llvmorg-14.0.3-install/bin/clang
#HOST_CXX_COMPILER_RUNTIME_LIB_DIR=/opt/llvm-github/llvm-project/llvmorg-14.0.3-install/lib
#
#CLANG_CXX_PROG="${HOST_CXX_COMPILER}"
#CLANG_CXX_RUNTIME_LIB_DIR="${HOST_CXX_COMPILER_RUNTIME_LIB_DIR}"

TVM_TRACKER_HOST="${TVM_TRACKER_HOST:-0.0.0.0}"
TVM_TRACKER_PORT="${TVM_TRACKER_PORT:-9192}"

# Must be one of the serial numbers provided by running 'adb devices -l'
# Note that some devices may already be in use, or may be incapable
# of running the software stack.
# The default serial number shown below was chosen arbitrarily from the
# list of currently valid serial numbers.
#ANDROID_SERIAL_NUMBER="${NEW_ANDROID_SERIAL_NUMBER:-c35b546f}"
ANDROID_SERIAL_NUMBER="${NEW_ANDROID_SERIAL_NUMBER:-192.168.10.138:5555}"


#-------------------------------------------------------------------------------
# Potentially customizable, but not worth the hassle...
#-------------------------------------------------------------------------------

ANDROID_TOOLCHAIN_CMAKE_FILE="${ANDROID_TOOLCHAIN_DIR}/build/cmake/android.toolchain.cmake"
LLVM_CONFIG_PROG="${LLVM_BIN_DIR}/llvm-config"

TVM_HEXAGON_API_SRC_DIR="${TVM_HOME}/apps/hexagon_api"
TVM_HEXAGON_API_BUILD_DIR="${TVM_HEXAGON_API_SRC_DIR}/build"
TVM_HEXAGON_API_OUTPUT_BINARY_DIR="${TVM_BUILD_DIR}/hexagon_api_output"

#-------------------------------------------------------------------------------
# Sanity checks
#-------------------------------------------------------------------------------

if ! [[ -f "${TVM_HOME}/CMakeLists.txt" ]]; then
    echo "This directory has no CMakeLists.txt file." >&2
    exit 1
fi

#-------------------------------------------------------------------------------
# Generate ctags-related files
#-------------------------------------------------------------------------------

CTAGS_FILE="${TVM_HOME}/.ctags"

cat > "${CTAGS_FILE}" << EOL
--languages=C,C++,CMake,Python
--kinds-C++=+lpvzLANUZ
--kinds-C=+lpxzL
--extras=+qr
--fields-Python={decorators}
--fields-C++=+{macrodef}{captures}{properties}{specialization}{template}
--fields-C=+{macrodef}{properties}
--fields=+EKRSilpr
-R
--exclude=${TVM_HEXAGON_API_BUILD_DIR}
--exclude=${TVM_BUILD_DIR}
EOL

#-------------------------------------------------------------------------------
# Generate the script for exercising the code once it's built...
#-------------------------------------------------------------------------------

MAIN_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_SCRIPT_NAME="$(basename -- "$0")"

# Files that this script generates, so that the user can invoke them sometime
# later...
ENV_SCRIPT="source-me.sh"

cat > "${ENV_SCRIPT}" << EOL
# THIS SCRIPT IS GENERATED BY ${MAIN_SCRIPT_NAME}.
#
# This script defines environment variables used by some/all of the other
# generated scripts. Its purpose is to help ensure consistency where it
# matters.

# These variables are exported (rather than simply being set) because the
# test_hexagon/rpc/test_launcher.py script uses them.
export TVM_HOME="${TVM_HOME}"
export TVM_TRACKER_HOST="${TVM_TRACKER_HOST}"
export TVM_TRACKER_PORT="${TVM_TRACKER_PORT}"
export HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN_TOOLS_DIR}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_DIR}"
export ANDROID_SERIAL_NUMBER="${ANDROID_SERIAL_NUMBER}"

# Prepend PYTHONPATH
if [[ ! -z "\${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${TVM_HOME}/python:\${PYTHONPATH}"
else
    export PYTHONPATH="${TVM_HOME}/python"
fi

# Append LD_LIBRARY_PATH
if [[ ! -z "\${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:${CLANG_CXX_RUNTIME_LIB_DIR}:${CXX_STDLIB_DIR}"
else
    export LD_LIBRARY_PATH="${LLVM_LIB_DIR}:${CXX_STDLIB_DIR}"
fi

alias hextvm_tracker_start='"${PYTHON3_CMD}" -m tvm.exec.rpc_tracker --host "\${TVM_TRACKER_HOST}" --port "\${TVM_TRACKER_PORT}"'

alias hextvm_tracker_query='"${PYTHON3_CMD}" -m tvm.exec.query_rpc_tracker --host "\${TVM_TRACKER_HOST}" --port "\${TVM_TRACKER_PORT}"'

alias hextvm_ctags='ctags "--options=${CTAGS_FILE}" "${TVM_HOME}" "${HEXAGON_SDK_DIR}/incs" "${HEXAGON_TOOLCHAIN_TOOLS_DIR}/include/iss"'
EOL

#-------------------------------------------------------------------------------
# Setup the source trees...
#-------------------------------------------------------------------------------

cd "${TVM_HOME}"

#-------------------------------------------------------------------------------
# Configure and build TVM Hexagon API server...
#-------------------------------------------------------------------------------

if [[ ! -z "\${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CLANG_CXX_RUNTIME_LIB_DIR}:${CXX_STDLIB_DIR}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LLVM_LIB_DIR}:${CXX_STDLIB_DIR}"
fi

mkdir -p "${TVM_HEXAGON_API_BUILD_DIR}"
cd "${TVM_HEXAGON_API_BUILD_DIR}"

    #-DCMAKE_CXX_COMPILER="${HOST_CXX_COMPILER}" \
    #-DCMAKE_C_COMPILER="${HOST_C_COMPILER}" \
    #-DCMAKE_CXX_COMPILER=clang++-10 \
    #-DCMAKE_C_COMPILER=clang-10 \

"${CMAKE_PROG}" -DUSE_ANDROID_TOOLCHAIN="${ANDROID_TOOLCHAIN_CMAKE_FILE}" \
    -G "${BUILD_SYSTEM}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" \
    -DCMAKE_CXX_COMPILER="${CLANG_CXX_PROG}" \
    -DCMAKE_C_COMPILER="${CLANG_C_PROG}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${CLANG_CXX_RUNTIME_LIB_DIR}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${CLANG_CXX_RUNTIME_LIB_DIR}" \
    -DANDROID_PLATFORM=android-28 \
    -DANDROID_ABI=arm64-v8a \
    -DUSE_HEXAGON_ARCH=v69 \
    -DUSE_HEXAGON_SDK="${HEXAGON_SDK_DIR}/" \
    -DUSE_HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN_TOOLS_DIR}" \
    -DUSE_OUTPUT_BINARY_DIR="${TVM_HEXAGON_API_OUTPUT_BINARY_DIR}" \
    -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
    ${TVM_EXTRA_CMAKE_ARGS:-} \
    "${TVM_HEXAGON_API_SRC_DIR}"

${BUILD_CMD}

#-------------------------------------------------------------------------------
# Configure and build TVM...
#-------------------------------------------------------------------------------

mkdir -p "${TVM_BUILD_DIR}"
cd "${TVM_BUILD_DIR}"

"${CMAKE_PROG}" \
      -DUSE_CPP_RPC=OFF \
      -G "${BUILD_SYSTEM}" \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" \
      -DCMAKE_CXX_COMPILER="${CLANG_CXX_PROG}" \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
      -DCMAKE_CXX_FLAGS_DEBUG="-ggdb -O0" \
      -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${CLANG_CXX_RUNTIME_LIB_DIR}" \
      -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
      -DUSE_LLVM="${LLVM_CONFIG_PROG}" \
      -DUSE_HEXAGON_SDK="${HEXAGON_SDK_DIR}" \
      -DUSE_HEXAGON_ARCH=v69 \
      -DUSE_HEXAGON=on \
      ${TVM_EXTRA_HEXAGON_CMAKE_ARGS:-} \
      "${TVM_HOME}"

${BUILD_CMD}

# This assumes Universal Ctags.
if $(which ctags >/dev/null); then
    ctags "--options=${CTAGS_FILE}" "${TVM_HOME}" "${HEXAGON_SDK_DIR}/incs" "${HEXAGON_TOOLCHAIN_TOOLS_DIR}/include/iss"
fi

