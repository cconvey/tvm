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

""" Various tests related to the (WIP) support for having
one PrimFunc call another PrimFunc within the same IRModule.
"""

from typing import List

# import pytest
import numpy as np

import tvm
import tvm.testing
import tvm.script
from tvm.script import tir as T

from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon import allocate_hexagon_array
from .infrastructure import get_hexagon_target

# --------------------------------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------------------------------

# The shape of the original (unsplit) tensors.
# We assume that each shape describes a non-empty 2D tensor.
ORIGINAL_SHAPE = tvm.testing.parameter(
    # degenerate cases...
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    # arbitrary, provided for variety
    [5, 3],
    [3, 5],
)

DTYPE = tvm.testing.parameter("int8")

# --------------------------------------------------------------------------------------------------
# Helper functions / definitions...
# --------------------------------------------------------------------------------------------------

HEXAGON_TARGET_ = get_hexagon_target("v69")

ENTRY_PRIMFUNC_NAME_ = "main"


def get_reference_input_tensor_(shape: list, dtype: str) -> np.array:
    assert len(shape) == 2

    a = np.ndarray(shape, dtype=dtype)
    np_dtype = a.dtype

    if np_dtype.kind in ["i", "u"]:
        # We allow overflow for integer types because it tends to be well-behaved
        # and well-understood...
        min_value = np.iinfo(np_dtype).min
        max_value = np.iinfo(np_dtype).max

        next_value = min_value

        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i, j] = next_value
                next_value += 1

    elif np_dtype.kind == "f":
        # NOTE: For simplicity, we avoid test data that that require
        # well-defined behavior on floating-point overflow.
        # But it may be reasonable to test that in the future.
        min_value = np.finfo(np_dtype).min
        max_value = np.finfo(np_dtype).max

        min_input_value = min_value / 2.0 + 1
        max_input_value = max_value / 2.0 - 2
        delta = (max_input_value - min_input_value) / (shape[0] * shape[1])

        next_value = min_input_value

        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i, j] = next_value
                next_value += delta

    else:
        assert False, f"Unexpected data type: {np_dtype}"

    return a


def get_reference_output_tensor_(shape: list, dtype: str) -> np.array:
    return get_reference_input_tensor_(shape, dtype) + 1


def evaluate_ir_module_(
    hexagon_session: Session, shape: List, dtype: str, ir_mod: tvm.ir.module.IRModule
) -> np.array:
    reference_input_np = get_reference_input_tensor_(shape, dtype)
    # reference_input_np = np.array([[11], [20], [31]], dtype="int8")
    # reference_input_np = np.full(shape, -1, dtype="int8")

    reference_output_np = get_reference_output_tensor_(shape, dtype)
    # reference_output_np = reference_input_np + 1

    hexagon_mod_local = tvm.build(
        ir_mod,
        target=get_hexagon_target("v69"),
        name=ENTRY_PRIMFUNC_NAME_,
    )

    hexagon_mod_remote = hexagon_session.load_module(hexagon_mod_local)

    input_data = allocate_hexagon_array(
        hexagon_session.device,
        data=reference_input_np,
    )

    output_data = allocate_hexagon_array(
        hexagon_session.device,
        tensor_shape=reference_output_np.shape,
        dtype=reference_output_np.dtype,
        data=np.full(shape, 0, dtype="int8"),
    )

    print(f"reference_input_np = {reference_input_np}")
    print(f"reference_output_np = {reference_output_np}")
    print(f"input_data = {input_data}")
    print(f"output_data = {output_data}")

    # breakpoint()

    hexagon_mod_remote(input_data, output_data)

    output_data_np = output_data.numpy()
    input_data_np = input_data.numpy()

    print(f"input_data_np = {input_data_np}")
    print(f"output_data_np = {output_data_np}")
    tvm.testing.assert_allclose(reference_output_np, output_data_np)


# --------------------------------------------------------------------------------------------------
# Test cases...
# --------------------------------------------------------------------------------------------------


@tvm.testing.requires_hexagon
def test_baseline(
    hexagon_session: Session, ORIGINAL_SHAPE: List, DTYPE: str
) -> tvm.ir.module.IRModule:
    dim0_size, dim1_size = ORIGINAL_SHAPE

    @tvm.script.ir_module
    class AddOneBaseline:
        """
        Provides "add-one" functionality in a single, traditional PrimFunc.
        Used as a baseline for comparison / validation with other approaches.
        I.e., approaches that use various aspects of PrimFunc slicing and/or
        one PrimFunc calling into another.
        """

        # pylint: disable=no-self-argument,invalid-name,missing-function-docstring
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr({"global_symbol": "main", "tir.noalias": True})

            A = T.match_buffer(a, ORIGINAL_SHAPE, dtype=DTYPE)
            B = T.match_buffer(b, ORIGINAL_SHAPE, dtype=DTYPE)

            for i in range(dim0_size):
                for j in range(dim1_size):
                    B[i, j] = A[i, j] + T.cast(1, DTYPE)

        # pylint: enable=no-self-argument,invalid-name,missing-function-docstring

    evaluate_ir_module_(hexagon_session, ORIGINAL_SHAPE, DTYPE, AddOneBaseline)


@tvm.testing.requires_hexagon
def test_pass_pointers(
    hexagon_session: Session, ORIGINAL_SHAPE: List, DTYPE: str
) -> tvm.ir.module.IRModule:
    dim0_size, dim1_size = ORIGINAL_SHAPE

    tile_shape = (dim1_size,)

    assert type(dim0_size) == int

    @tvm.script.ir_module
    class AddOnePassPointers:
        # pylint: disable=no-self-argument,invalid-name,missing-function-docstring
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.is_entry_func": True})

            A = T.match_buffer(a, ORIGINAL_SHAPE, dtype=DTYPE)
            B = T.match_buffer(b, ORIGINAL_SHAPE, dtype=DTYPE)

            for i in range(dim0_size):
                T.call_extern("", "callee", A.data, B.data, i)

        @T.prim_func
        def callee(a_data: T.Ptr[T.int8], b_data: T.Ptr[T.int8], i: T.int32):
            T.func_attr(
                {
                    "global_symbol": "callee",
                    "tir.noalias": True,
                    "from_legacy_te_schedule": False,
                    "calling_conv": 3,
                }
            )

            A_tile = T.buffer_decl(tile_shape, DTYPE, a_data, elem_offset=dim1_size * i)
            B_tile = T.buffer_decl(tile_shape, DTYPE, b_data, elem_offset=dim1_size * i)

            for j in range(dim1_size):
                B_tile[j] = A_tile[j] + T.int8(1)

        # pylint: enable=no-self-argument,invalid-name,missing-function-docstring

    # dump_passes = False
    dump_passes = True
    if dump_passes:
        from .lunderberg_tvm_instrument import PrintTransformSequence

        with tvm.transform.PassContext(instruments=[PrintTransformSequence()]):
            evaluate_ir_module_(hexagon_session, ORIGINAL_SHAPE, DTYPE, AddOnePassPointers)
    else:
        evaluate_ir_module_(hexagon_session, ORIGINAL_SHAPE, DTYPE, AddOnePassPointers)


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tvm.testing.main()
