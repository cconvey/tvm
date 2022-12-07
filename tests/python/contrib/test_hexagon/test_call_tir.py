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

from typing import List
import tvm
import tvm.testing
from tvm.script import tir as T
import pytest


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

import os
import os.path
import sys
import tempfile

import numpy as np
import pytest

import tvm.script
import tvm.testing
from tvm.script import tir as T

from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon import allocate_hexagon_array

from .infrastructure import get_hexagon_target

# from .benchmark_util import get_benchmark_id

# --------------------------------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------------------------------

# The shape of the original (unsplit) tensors.
# We assume that each shape describes a non-empty 2D tensor.
original_shape = tvm.testing.parameter(
    # degenerate cases...
    [1, 1],
    [1, 3],
    [3, 1],
    # arbitrary but small..
    [3, 5],
)

dtype = tvm.testing.parameter("int8")

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
    reference_output_np = get_reference_output_tensor_(shape, dtype)

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
    )

    hexagon_mod_remote(input_data, output_data)

    output_data_np = output_data.numpy()
    tvm.testing.assert_allclose(reference_output_np, output_data_np)


# --------------------------------------------------------------------------------------------------
# Test cases...
# --------------------------------------------------------------------------------------------------


@tvm.testing.requires_hexagon
def test_baseline(
    hexagon_session: Session, original_shape: List, dtype: str
) -> tvm.ir.module.IRModule:
    dim0_size, dim1_size = original_shape

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

            A = T.match_buffer(a, original_shape, dtype=dtype)
            B = T.match_buffer(b, original_shape, dtype=dtype)

            for i in range(dim0_size):
                for j in range(dim1_size):
                    B[i, j] = A[i, j] + T.cast(1, dtype)

        # pylint: enable=no-self-argument,invalid-name,missing-function-docstring

    evaluate_ir_module_(hexagon_session, original_shape, dtype, AddOneBaseline)


if __name__ == "__main__":
    tvm.testing.main()
