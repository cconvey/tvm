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

class TestMaxPool2D:
    dtype = tvm.testing.parameter("int8")

    #N = tvm.testing.parameter(1)
    #H = tvm.testing.parameter(128)
    #W = tvm.testing.parameter(128)
    #C = tvm.testing.parameter(64)

    # Exactly one crouton
    N = tvm.testing.parameter(1)
    H = tvm.testing.parameter(8)
    W = tvm.testing.parameter(8)
    C = tvm.testing.parameter(32)

    #kernel = tvm.testing.parameter((3, 3))
    kernel = tvm.testing.parameter((1, 1))

    stride = tvm.testing.parameter((1, 1))
    dilation = tvm.testing.parameter((1, 1))
    padding = tvm.testing.parameter((0, 0, 0, 0))

    @tvm.testing.requires_hexagon
    def test_maxpool2d_nhwc(
        self,
        N, # 1
        H, # 128
        W, # 128
        C, # 64
        dtype,
        kernel, # (3,3)
        stride, # (1,1)
        dilation, # (1,1)
        padding, # (0,0,0,0)
        hexagon_session: Session,
        ):
        data = te.placeholder((N, H, W, C), dtype=dtype) # data.shape = [1, 128, 128, 64]
        output = topi.nn.pool2d(data, kernel, stride, dilation, padding, "max", layout="NHWC")  # output: tvm.te.tensor.Tensor ; output.shape = [1,126,126,64]

        #@T.prim_func
        #def func(var_placeholder: T.handle, tensor: T.Buffer[2048, "int8"]) -> None:
        #    # function attr dict
        #    T.func_attr({"global_symbol": "main", "tir.noalias": True})
        #    placeholder = T.match_buffer(var_placeholder, [1, 2048], dtype="int8", axis_separators=[1])
        #    T.preflattened_buffer(placeholder, [1, 1, 1, 1, 8, 8, 32], dtype="int8", data=placeholder.data, axis_separators=[4])
        #    T.preflattened_buffer(tensor, [1, 8, 8, 32], dtype="int8", data=tensor.data)
        #    # body
        #    for i1, i2, i3 in T.grid(8, 8, 32):
        #        cse_var_1: T.int32 = i1 * 256 + i2 * 32 + i3
        #        tensor[cse_var_1] = T.int8(-128)
        #        tensor[cse_var_1] = T.max(tensor[cse_var_1], placeholder[0, cse_var_1])
        #primfunc = func

        # Disabled because we're copy-pasting TVMScript
        primfunc = te.create_prim_func([data, output])  # type(primfunc) = tvm.tir.function.PrimFunc

        with open('out-2a.txt', 'w') as f:
            f.write(str(primfunc))
        with open('out-2b.txt', 'w') as f:
            f.write(str(primfunc.script()))

        sch = tir.Schedule(primfunc, debug_mask="all") # tvm.tir.schedule.schedule.Schedule

        with open('out-3a.txt', 'w') as f:
            f.write(str(sch.mod['main']))
        with open('out-3b.txt', 'w') as f:
            f.write(str(sch.mod['main'].script()))

        # Line 74 in Chris's script
        # Disabled while we're using TVMScript
        sch.transform_layout(block="tensor", buffer="placeholder", index_map=int8_nhwc_8h8w32c)

        foo = 'with-axis-separator'

        with open(f'out-4-{foo}.txt', 'w') as f:
            f.write(str(sch.mod['main']))
            f.write(str(sch.mod['main'].script()))

        with open(f'out-5-{foo}.txt', 'w') as f:
            foo = tvm.lower(sch.mod, [data, output,])['main']
            f.write(str(foo))
            f.write(str(foo.script()))


        # compute : tvm.tir.schedule.schedule.BlockRV
        mod = sch.mod

        print(mod["main"].script())
        print(tvm.lower(mod))

        #return

        target_hexagon = tvm.target.hexagon("v69", link_params=True)
        func = tvm.build(mod, target=tvm.target.Target(target_hexagon, host=target_hexagon))
        func.save('benchmark_maxpool2d_hexagon.so')
        mod = hexagon_session.load_module(func)

        a_np = np.random.randint(low=-128, high=127, size=(N, H, W, C), dtype=np.int8)

        # Random is overrated while debugging...
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        a_np[n,h,w,c] = 42


        ref_output = testing.poolnd_python(
            #a_np.astype("int32"),
            a_np.astype("int8"), # ????
            kernel,
            stride,
            dilation,
            padding[0:2],
            padding[2:],
            pool_type="max",
            dtype="int32",
            layout="NHWC",
        ).astype("int8")


	#breakpoint()

        # Line 105 in Chris' script.
        a_transformed = a_np.reshape(N, H // 8, 8, W // 8, 8, C // 32, 32).transpose(
            0, 1, 3, 5, 2, 4, 6
        )

        #input_shape = [1,1,1,1,8,8,32]
        #output_shape = [1,8,8,32]

        packed_input_shape = get_packed_shape([N,H,W,C])

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

        #breakpoint()

        print('AAAAA: a_hexagon.numpy()[0,0,0,0,0,0,0]={}'.format(a_hexagon.numpy()[0,0,0,0,0,0,0]))

        a_hexagon.copyfrom(a_transformed)

        #time.sleep(5)
        print('BBBBB: a_hexagon.numpy()[0,0,0,0,0,0,0]={}'.format(a_hexagon.numpy()[0,0,0,0,0,0,0]))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
