# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import os.path
import sys
import pytest
import numpy as np
import logging
import tempfile

import tvm.testing
import tvm.script
from tvm.script import tir as T
from tvm import te
from tvm.contrib.hexagon.build import HexagonLauncherRPC
from .benchmark_util import BenchmarksTable

RPC_SERVER_PORT = 7070

# This is a fixed detail of the v68 architecture.
HVX_VECTOR_BYTES = 128

# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.

_BT = BenchmarksTable()

_CSV_COLUMN_ORDER = [
    # Identifies which TE-compute / TIRScript is used as the basis for the
    # benchmarked primfunc. Only needs to be meaningful to humans.
    "basic_kernel",
    # The tensors 'element type
    "dtype",
    # When applicable, indicates the particular variation of schedules
    # apply by the Python code. Decoding this may require looking at this
    # script's source code.
    "sched_type",
    # The memory location of the tensors used during the execution of
    # the primfunc.  We currently assume just one location.
    # This will likely need to be generalized as we add more sophisticated
    # primfuncs.
    "mem_scope",
    # For primfuncs that treat tensor buffers as collections of 1D vectors,
    # this is the number of vectors in each tensor.
    # This will likely need to be generalized as we add more sophisticated
    # primfuncs.
    "num_vectors_per_tensor",
    # Reserved columns defined by the BenchmarksTable class.
    "row_status",
    "timings_min_usecs",
    "timings_max_usecs",
    "timings_median_usecs",
    "timings_mean_usecs",
    "timings_stddev_usecs",
    # For benchmarks that produce files on the host file system, this indicates
    # their location. Useful for post-mortem investigation of benchmark results.
    "host_files_dir",
    # Miscellaneous comments about the benchmark.
    "comments",
]

_HOST_OUTPUT_DIR = tempfile.mkdtemp()

print("-" * 80)
print("OUTPUT DIRECTORY: {}".format(_HOST_OUTPUT_DIR))
print("-" * 80)
print()


def _get_benchmark_id(keys_dict):
    """
    Given a dictionary with the distinguishing characteristics of a particular benchmark
    line item, compute a string that uniquely identifies the benchmark.

    The returned string:
    - is a valid directory name on the host's file system
    - should be easy for humans to parse

    Note that the insertion order for `keys_dict` does affect the computed name.
    """
    return "-".join([f"{k}:{v}" for k, v in keys_dict.items()])


def _get_benchmark_decription(keys_dict):
    """
    Similar to `_get_benchmark_id`, but the focus is on human-readability.

    The returned string contains no line-breaks, but may contain spaces and
    other characters that make it unsuitable for use as a filename.
    """
    return " ".join([f"{k}={v}" for k, v in keys_dict.items()])


@tvm.testing.requires_hexagon
def test_elemwise_add_tvmcript(hexagon_launcher: HexagonLauncherRPC):
    """
    Similar to `test_elemwise_add_te`, but starting with TensorScript rather than
    Tensor Expressions.
    """

    # Create and benchmark a single primfunc.
    # If an unexpected problem occurs, raise an exception.  Otherwise add a row of output to 'bt'.
    def test_one_config(dtype, mem_scope, num_vectors_per_tensor):
        basic_kernel = "ewise-tvmscript-1"

        # The distinguishing characteristics of this benchmark line item.
        keys_dict = {
            "basic_kernel": basic_kernel,
            "dtype": dtype,
            "mem_scope": mem_scope,
            "num_vectors_per_tensor": num_vectors_per_tensor,
        }

        host_files_dir_name = _get_benchmark_id(keys_dict)
        desc = _get_benchmark_decription(keys_dict)

        print(f"CONFIGURATION: {desc}")

        if num_vectors_per_tensor == 2048 and mem_scope == "global.vtcm":
            _BT.record_skip(**keys_dict, comments="Expect to exceed VTCM budget.")
            return

        host_files_dir = os.path.join(_HOST_OUTPUT_DIR, host_files_dir_name)
        os.mkdir(host_files_dir)

        dtype_bits = tvm._ffi.runtime_ctypes.DataType(dtype).bits
        assert dtype_bits % 8 == 0
        dtype_bytes = dtype_bits // 8

        elem_per_hvx_vector = HVX_VECTOR_BYTES // dtype_bytes

        shape = [
            num_vectors_per_tensor,
            elem_per_hvx_vector,
        ]

        # TVMScript can reference simple Python variables, but it doesn't
        # curently support more complex Python expressions...
        dim0_size = shape[0]
        dim1_size = shape[1]
        dtype_str = str(dtype)

        @tvm.script.ir_module
        class MyModule:
            @T.prim_func
            def main(a: T.handle, b: T.handle, c: T.handle):
                # We exchange data between function by handles, which are similar to pointer.
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                # Create buffer from handles.
                A = T.match_buffer(a, shape, dtype=dtype_str)
                B = T.match_buffer(b, shape, dtype=dtype_str)
                C = T.match_buffer(c, shape, dtype=dtype_str)

                for i in range(dim0_size):
                    for j in range(dim1_size):
                        C[i, j] = A[i, j] + B[i, j]

        ir_module = MyModule

        A = tvm.te.placeholder(shape, dtype=dtype)
        B = tvm.te.placeholder(shape, dtype=dtype)
        C = tvm.te.placeholder(shape, dtype=dtype)

        module_for_ir_dump = tvm.lower(ir_module, [A, B, C], "elemwise_add")

        report_path = os.path.join(host_files_dir, "out.txt")
        with open(report_path, "w") as f:
            f.write("LOWERED IR MODULE:\n")
            f.write(str(module_for_ir_dump))
            f.write("\n")

            target_hexagon = tvm.target.hexagon("v68", link_params=True)
            func = tvm.build(
                # sched,
                ir_module,
                [A, B, C],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="elemwise_add",
            )

            host_dso_binary_path = os.path.join(host_files_dir, "test_binary.so")
            target_dso_binary_filename = "test_binary.so"

            func.save(host_dso_binary_path)
            print(f"SAVED BINARY TO HOST PATH: {host_dso_binary_path}")

            hexagon_launcher.upload(host_dso_binary_path, target_dso_binary_filename)

            try:
                with hexagon_launcher.start_session() as sess:
                    mod = hexagon_launcher.load_module(target_dso_binary_filename, sess)

                    host_numpy_A_data = np.ndarray(shape, dtype=dtype)
                    host_numpy_B_data = np.ndarray(shape, dtype=dtype)

                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            host_numpy_A_data[i, j] = i + j
                            host_numpy_B_data[i, j] = (i + 1) * (j + 1)

                    host_numpy_C_data_expected = host_numpy_A_data + host_numpy_B_data

                    A_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    A_data.copyfrom(host_numpy_A_data)

                    B_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    B_data.copyfrom(host_numpy_B_data)

                    C_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)

                    # NOTE: We may want to soften these numbers, depending on future findings.
                    timer = mod.time_evaluator("main", sess.device, number=10, repeat=1)
                    timing_result = timer(A_data, B_data, C_data)

                    print("TIMING RESULT: {}".format(timing_result))

                    # Verify that the computation actually happened, and produced the correct result.
                    result = C_data.numpy()
                    tvm.testing.assert_allclose(host_numpy_C_data_expected, result)

                    _BT.record_success(timing_result, host_files_dir=host_files_dir, **keys_dict)

            except Exception as err:
                print()
                print(f"FAILURE: See {report_path}")
                f.write("ERROR:\n")
                f.write(f"{err}\n")
                _BT.record_fail(
                    **keys_dict, host_files_dir=host_files_dir, comments=f"See {report_path}"
                )

    # -----------------------------------------------------------------------------------------------

    # Hexagon v69 allows more dtypes, but we're sticking with v68 for now.
    for dtype in [
        "int8",
    ]:

        for mem_scope in [
            "global",
        ]:

            # These numbers are fairly arbitrary, but they're meant to stress memory/caches to
            # various extents.
            for num_vectors_per_tensor in [
                1,
                16,
                64,
                512,
                2048,
            ]:

                print()
                test_one_config(dtype, mem_scope, num_vectors_per_tensor)

    print("-" * 80)
    print(f"OUTPUT DIRECTORY: {_HOST_OUTPUT_DIR}")
    print("-" * 80)
    print()

    tabular_output_filename = os.path.join(_HOST_OUTPUT_DIR, "benchmark-results.csv")
    with open(tabular_output_filename, "w") as csv_file:
        _BT.print_csv(csv_file, _CSV_COLUMN_ORDER)

    print(f"BENCHMARK RESULTS FILE: {tabular_output_filename}")

    _BT.print_csv(sys.stdout, _CSV_COLUMN_ORDER)

    if _BT.has_fail() > 0:
        pytest.fail("At least one benchmark configuration failed", pytrace=False)


@tvm.testing.requires_hexagon
def test_elemwise_add_te(hexagon_launcher: HexagonLauncherRPC):
    """
    Starting with an elementwise-add computation, try various schedules / optimizations to
    see the impact they have on performance.

    The main motivation for this test is to explore the relationship between these
    schedules / optimizations vs. how effectively the primfunc uses the Hexagon's
    HVX units.
    """

    # Create and benchmark a single primfunc.
    # If an unexpected problem occurs, raise an exception.  Otherwise add a row of output to 'bt'.
    def test_one_config(dtype, sched_type, mem_scope, num_vectors_per_tensor):
        basic_kernel = "elemwise-add-te"

        # The distinguishing characteristics of this benchmark line item.
        keys_dict = {
            "basic_kernel": basic_kernel,
            "dtype": dtype,
            "sched_type": sched_type,
            "mem_scope": mem_scope,
            "num_vectors_per_tensor": num_vectors_per_tensor,
        }

        host_files_dir_name = _get_benchmark_id(keys_dict)
        desc = _get_benchmark_decription(keys_dict)

        print(f"CONFIGURATION: {desc}")

        if num_vectors_per_tensor == 2048 and mem_scope == "global.vtcm":
            result = dict(keys_dict)
            _BT.record_skip(**keys_dict, comments="Expect to exceed VTCM budget.")
            return

        host_files_dir = os.path.join(_HOST_OUTPUT_DIR, host_files_dir_name)
        os.mkdir(host_files_dir)

        dtype_bits = tvm._ffi.runtime_ctypes.DataType(dtype).bits
        assert dtype_bits % 8 == 0
        dtype_bytes = dtype_bits // 8

        elem_per_hvx_vector = HVX_VECTOR_BYTES // dtype_bytes

        # Note!  We're providing the complete input tensor shapes now,
        # whereas the original code only reveals the exact shape when
        # about to call the kernel.

        shape = [
            num_vectors_per_tensor,
            elem_per_hvx_vector,
        ]

        A = tvm.te.placeholder(shape, dtype=dtype)
        B = tvm.te.placeholder(shape, dtype=dtype)
        C = tvm.te.compute(A.shape, lambda i, j: A[i, j] + B[i, j], name="C")

        sched = tvm.te.create_schedule(C.op)

        if sched_type == 1:
            pass
        elif sched_type == 2:
            sched[C].vectorize(C.op.axis[1])
        else:
            raise Exception("Unknown schedule type")

        # If we're using VTCM, we *must* add a transform_layout step to the schedule.
        # Otherwise the generated code will crash.
        # As of 2022-04-12 the crash does not provide a useful error message to the
        # host Python code.
        if mem_scope == "global.vtcm":
            for tensor in [A, B, C]:
                sched[tensor].transform_layout(lambda i, j: [i, te.AXIS_SEPARATOR, j])

        # This module is only created so humans can inspect its IR.
        module_for_ir_dump = tvm.lower(sched, [A, B, C], "foo")

        report_path = os.path.join(host_files_dir, "out.txt")
        with open(report_path, "w") as f:
            f.write("LOWERED IR MODULE:\n")
            f.write(str(module_for_ir_dump))
            f.write("\n")

            target_hexagon = tvm.target.hexagon("v68", link_params=True)
            func = tvm.build(
                sched,
                [A, B, C],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="elemwise_add",
            )

            host_dso_binary_path = os.path.join(host_files_dir, "test_binary.so")
            target_dso_binary_filename = "test_binary.so"

            func.save(host_dso_binary_path)
            print(f"SAVED BINARY TO HOST PATH: {host_dso_binary_path}")

            hexagon_launcher.upload(host_dso_binary_path, target_dso_binary_filename)

            try:
                with hexagon_launcher.start_session() as sess:
                    mod = hexagon_launcher.load_module(target_dso_binary_filename, sess)

                    host_numpy_A_data = np.ndarray(shape, dtype=dtype)
                    host_numpy_B_data = np.ndarray(shape, dtype=dtype)

                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            host_numpy_A_data[i, j] = i + j
                            host_numpy_B_data[i, j] = (i + 1) * (j + 1)

                    host_numpy_C_data_expected = host_numpy_A_data + host_numpy_B_data

                    A_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    A_data.copyfrom(host_numpy_A_data)

                    B_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    B_data.copyfrom(host_numpy_B_data)

                    C_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)

                    # NOTE: We may want to soften these numbers, depending on future findings.
                    timer = mod.time_evaluator("elemwise_add", sess.device, number=10, repeat=1)
                    timing_result = timer(A_data, B_data, C_data)

                    print("TIMING RESULT: {}".format(timing_result))

                    # Verify that the computation actually happened, and produced the correct result.
                    result = C_data.numpy()
                    tvm.testing.assert_allclose(host_numpy_C_data_expected, result)

                    _BT.record_success(timing_result, **keys_dict, host_files_dir=host_files_dir)

            except Exception as err:
                print()
                print(f"FAILURE: See {report_path}")
                f.write("ERROR:\n")
                f.write("{}\n".format(err))
                _BT.record_fail(
                    **keys_dict, host_files_dir=host_files_dir, comments=f"See {report_path}"
                )

    # -----------------------------------------------------------------------------------------------

    # Hexagon v69 allows more dtypes, but we're sticking with v68 for now.
    for dtype in [
        "int8",
    ]:

        # These numbers are only meaningful in the context of this script.
        for sched_type in [
            1,
            2,
        ]:

            for mem_scope in ["global", "global.vtcm"]:

                # These numbers are fairly arbitrary, but they're meant to stress memory/caches to
                # various extents.
                for num_vectors_per_tensor in [
                    1,
                    16,
                    64,
                    512,
                    2048,
                ]:

                    test_one_config(dtype, sched_type, mem_scope, num_vectors_per_tensor)

    tabular_output_filename = os.path.join(_HOST_OUTPUT_DIR, "benchmark-results.csv")
    with open(tabular_output_filename, "w") as csv_file:
        _BT.print_csv(csv_file, _CSV_COLUMN_ORDER)
    print(f"BENCHMARK RESULTS FILE: {tabular_output_filename}")

    _BT.print_csv(sys.stdout, _CSV_COLUMN_ORDER)

    if _BT.has_fail() > 0:
        pytest.fail("At least one benchmark configuration failed", pytrace=False)
