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

import sys
import pytest
import numpy as np

import tvm.testing
from tvm import te, topi, tir
from tvm.topi import testing
from tvm.script import tir as T
from tvm.tir import IndexMap
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.hexagon.session import Session


from .infrastructure import allocate_hexagon_array


# This is the input-tensor layout that influenced the TVMScript below.
# def int8_nhwc_8h8w32c(n, h, w, c):
#    return [
#        n,
#        h // 8,
#        w // 8,
#        c // 32,
#        IndexMap.AXIS_SEPARATOR,
#        h % 8,
#        w % 8,
#        c % 32,
#    ]

# maxpool input shape is exactly one "chunk".
N = 1
H = 8
W = 8
C = 32
dtype = "int8"

input_shape_4d = [N, H, W, C]
input_shape_7d = [
    1,  # N
    1,  # H // 8
    1,  # W // 8
    1,  # C // 32
    8,  # H % 8
    8,  # W % 8
    32,  # C % 32
]
output_shape_4d = input_shape_4d


class TestMaxPool2D:
    @tvm.testing.requires_hexagon
    def test_maxpool2d_nhwc(
        self,
        hexagon_session: Session,
    ):

        # Obtain the TIR primfunc...
        @T.prim_func
        def func(var_placeholder: T.handle, tensor: T.Buffer[2048, "int8"]) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            placeholder = T.match_buffer(
                var_placeholder, [1, 2048], dtype="int8", axis_separators=[1]
            )
            T.preflattened_buffer(
                placeholder,
                [1, 1, 1, 1, 8, 8, 32],
                dtype="int8",
                data=placeholder.data,
                axis_separators=[4],
            )
            T.preflattened_buffer(tensor, [1, 8, 8, 32], dtype="int8", data=tensor.data)
            # body
            for i1, i2, i3 in T.grid(8, 8, 32):
                # THE LOOP BODY PRODUCED BY OUR NORMAL FLOW.  CRASHES.
                cse_var_1: T.int32 = i1 * 256 + i2 * 32 + i3
                tensor[cse_var_1] = T.int8(-128)
                tensor[cse_var_1] = T.max(tensor[cse_var_1], placeholder[0, cse_var_1])

                # ALTERNATIVE BODIES I TRIED, AND HOW THEY TURNED OUT:

                # -----------------------------------------------------------------------------------------
                # PATTERN #1: Seems like bare '42' is treated as a 4-byte r-value, but
                #    T.int8(42) is treated like a 1-byte r-value?
                # However, this may be irrelevant because this doesn't show up on our TVMScript.
                # -----------------------------------------------------------------------------------------

                # OK:    tensor[0] = 42
                # OK:    tensor[2044] = 42
                # OK:    tensor[2045] = T.int8(42)
                # CRASH: tensor[2045] = 42
                # OK: tensor[2044] = tensor[0]
                # OK: tensor[2047] = tensor[0]

                # -----------------------------------------------------------------------------------------
                # PATTERN #2: Any indexing into placeholder[...] causes a crash.
                # -----------------------------------------------------------------------------------------

                # CRASH: tensor[cse_var_1] = 42
                # CRASH: tensor[cse_var_1] = T.max(T.int8(42), T.int8(placeholder[0, cse_var_1]))
                # CRASH: tensor[cse_var_1] = T.max(T.int8(42), placeholder[0, cse_var_1])
                # CRASH: tensor[cse_var_1] = T.max(tensor[cse_var_1], placeholder[0, cse_var_1])

                # CRASH:
                #    foo_2: T.int8 = placeholder[0, cse_var_1]
                #    tensor[cse_var_1] = T.max(T.int8(42), foo_2)

                # CRASH: placeholder[0, cse_var_1] = T.int8(tensor[cse_var_1])
                # CRASH: placeholder[0, cse_var_1] = tensor[cse_var_1]
                # CRASH: tensor[0] = placeholder[0, 0]
                # CRASH: tensor[2044] = placeholder[0, 2044]
                # CRASH: tensor[cse_var_1] = T.int8( T.max(T.int8(42), placeholder[0, cse_var_1]) )
                # CRASH: tensor[cse_var_1] = placeholder[0, cse_var_1]

                # OK:    tensor[cse_var_1] = T.int8(42)
                # OK:    tensor[cse_var_1] = T.max(T.int8(42), T.int8(43))
                # OK:    tensor[cse_var_1] = T.max(tensor[cse_var_1], T.int8(43))

        tir_primfunc = func

        # Compile the TIR primfunc into v69 object code...
        # NOTE: v68 crashes too.
        target_hexagon = tvm.target.hexagon("v69", link_params=True)
        hexagon_primfunc = tvm.build(
            tir_primfunc, target=tvm.target.Target(target_hexagon, host=target_hexagon)
        )

        # Send the v69 object code over to the Android/Hexagon device, and create a host-side
        # proxy module...
        hexagon_mod = hexagon_session.load_module(hexagon_primfunc)

        # Allocate the primfunc's input/output tensors on the Hexagon's DDR, and create host-side proxy tensors...
        # a_hexagon = tvm.nd.empty(
        #     input_shape_7d, dtype="int8", device=hexagon_session.device, mem_scope="global"
        # )
        a_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=input_shape_7d,
            axis_separators=[4],
            dtype="int8",
            mem_scope="global.vtcm",
        )

        # c_hexagon = tvm.nd.empty(
        #     output_shape_4d, dtype="int8", device=hexagon_session.device, mem_scope="global"
        # )
        c_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=output_shape_4d,
            axis_separators=[],
            dtype="int8",
            mem_scope="global.vtcm",
        )

        # Populate the primfunc's input tensor, which resides in Hexagon DDR, with zeros.
        # NOTE: We still crash when these lines are uncommented.
        # a_host = np.zeros(input_shape_7d).astype("int8")
        # a_hexagon.copyfrom(a_host)

        # Execute the primfunc on Hexagon.
        hexagon_mod(a_hexagon, c_hexagon)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
