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

from .infrastructure import allocate_hexagon_array, get_packed_shape

import copy
import time

def int8_nhwc_8h8w32c(n, h, w, c):
    shape = [
        n,
        h // 8,
        w // 8,
        c // 32,
        h % 8,
        w % 8,
        c % 32,
    ]

    return shape

def as_python_int(x):
    if isinstance(x, tvm.tir.expr.IntImm):
        return x.value
    elif isinstance(x, np.integer):
        return x.item()
    elif isinstance(x, int):
        return x

    raise Exception(f"Don't have a converter for type {type(x)}")

class TestMaxPool2D:
    dtype = tvm.testing.parameter("int8")

    N = tvm.testing.parameter(1)
    H = tvm.testing.parameter(128)
    W = tvm.testing.parameter(128)
    C = tvm.testing.parameter(64)

    # Exactly one crouton
    #N = tvm.testing.parameter(1)
    #H = tvm.testing.parameter(8)
    #W = tvm.testing.parameter(8)
    #C = tvm.testing.parameter(32)

    #kernel = tvm.testing.parameter((3, 3))
    kernel = tvm.testing.parameter((1, 1))

    #stride = tvm.testing.parameter((1, 1))
    #dilation = tvm.testing.parameter((1, 1))
    #padding = tvm.testing.parameter((0, 0, 0, 0))

    @tvm.testing.requires_hexagon
    def test_maxpool2d_nhwc(
        self,
        N, # 1
        H, # 128
        W, # 128
        C, # 64
        dtype,
        hexagon_session: Session,
        ):

        #N_int = N.astype('int')
        #H_int = H.astype('int')
        #W_int = W.astype('int')
        #C_int = C.astype('int')

        packed_input_shape_tir = get_packed_shape([N, H, W, C])
        packed_input_shape_python = [ as_python_int(x) for x in packed_input_shape_tir ]

        target_hexagon = tvm.target.hexagon("v69", link_params=True)

        a_np = np.random.randint(low=-128, high=127, size=(N, H, W, C), dtype=np.int8)

        # Random is overrated while debugging...
        a_np[:] = 42

        # Line 105 in Chris' script.

        # Debugging:
        #  OK: shape=[1,1], np array not reshaped, not reshaped, not transposed

        #a_np_7d = np.zeros([1, 1, 1, 1, 8, 8, 32]).astype('int8')
        ###breakpoint()
        a_np_7d = np.ndarray(shape=tuple(packed_input_shape_python), dtype='int8') #
        a_np_7d[:] = 42

        a_transformed = a_np.reshape(N, H // 8, 8, W // 8, 8, C // 32, 32).transpose(
            0, 1, 3, 5, 2, 4, 6
        )

        a_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=get_packed_shape([N,H,W,C]),
            axis_separators=[4],
            dtype="int8",
            mem_scope="global.vtcm",
        )

        c_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=[N,H,W,C],
            axis_separators=[],
            dtype="int8",
            mem_scope="global.vtcm",
        )


        #a_hexagon = allocate_hexagon_array(
        #    hexagon_session.device,
        #    #tensor_shape=get_packed_shape([N,H,W,C]),
        #    tensor_shape=packed_input_shape_python,
        #    axis_separators=[4],
        #    dtype="int8",
        #    mem_scope="global.vtcm",
        #)

        print('A: a_hexagon.numpy()[0,0,0,0,0,0,0] = {}\n'.format(a_hexagon.numpy()[0,0,0,0,0,0,0]))

        a_hexagon.copyfrom(a_transformed)

        #time.sleep(5)

        print('B: a_hexagon.numpy()[0,0,0,0,0,0,0] = {}\n'.format(a_hexagon.numpy()[0,0,0,0,0,0,0]))

        #time.sleep(5)

        #print('C: a_hexagon.numpy()[0,0,0,0,0,0,0] = {}\n'.format(a_hexagon.numpy()[0,0,0,0,0,0,0]))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

