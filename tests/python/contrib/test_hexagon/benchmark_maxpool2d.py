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


USE_AXIS_SEPARATOR = True
#USE_AXIS_SEPARATOR = False

def int8_nhwc_8h8w32c(n, h, w, c):
    if USE_AXIS_SEPARATOR:
        return [
            n,
            h // 8,
            w // 8,
            c // 32,
            IndexMap.AXIS_SEPARATOR,
            h % 8,
            w % 8,
            c % 32,
        ]
    else:
        return [
            n,
            h // 8,
            w // 8,
            c // 32,
            #IndexMap.AXIS_SEPARATOR,
            h % 8,
            w % 8,
            c % 32,
        ]

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
        #output = topi.nn.pool2d(data, kernel, stride, dilation, padding, "max", layout="NHWC")  # output: tvm.te.tensor.Tensor ; output.shape = [1,126,126,64]

        @T.prim_func
        def func(var_placeholder: T.handle, tensor: T.Buffer[2048, "int8"]) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            placeholder = T.match_buffer(var_placeholder, [1, 2048], dtype="int8", axis_separators=[1])
            T.preflattened_buffer(placeholder, [1, 1, 1, 1, 8, 8, 32], dtype="int8", data=placeholder.data, axis_separators=[4])
            T.preflattened_buffer(tensor, [1, 8, 8, 32], dtype="int8", data=tensor.data)
            # body
            for i1, i2, i3 in T.grid(8, 8, 32):
                cse_var_1: T.int32 = i1 * 256 + i2 * 32 + i3
                tensor[cse_var_1] = T.int8(-128)
                tensor[cse_var_1] = T.max(tensor[cse_var_1], placeholder[0, cse_var_1])
        primfunc = func

        #with open('out-1.txt', 'w') as f:
        #    f.write(str(tvm.lower(tvm.te.create_schedule(output.op), [data, output,], simple_mode=True)))

        # Disabled because we're copy-pasting TVMScript
        #primfunc = te.create_prim_func([data, output])  # type(primfunc) = tvm.tir.function.PrimFunc

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
        # sch.transform_layout(block="tensor", buffer="placeholder", index_map=int8_nhwc_8h8w32c)

        if USE_AXIS_SEPARATOR:
            foo = 'with-axis-separator'
        else:
            foo = 'sans-axis-separator'

        with open(f'out-4-{foo}.txt', 'w') as f:
            f.write(str(sch.mod['main']))
            f.write(str(sch.mod['main'].script()))

        #with open(f'out-5-{foo}.txt', 'w') as f:
        #    foo = tvm.lower(sch.mod, [data, output,])['main']
        #    f.write(str(foo))
        #    f.write(str(foo.script()))


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
        ref_output = testing.poolnd_python(
            a_np.astype("int32"),
            kernel,
            stride,
            dilation,
            padding[0:2],
            padding[2:],
            pool_type="max",
            dtype="int32",
            layout="NHWC",
        ).astype("int8")

	# Do we actually need c_np?
        c_np = np.zeros(ref_output.shape).astype("int8")

	breakpoint()

        # Line 105 in Chris' script.
        a_transformed = a_np.reshape(N, H // 8, 8, W // 8, 8, C // 32, 32).transpose(
            0, 1, 3, 5, 2, 4, 6
        )

        #a_transformed_step1 =  a_np.reshape(N, H // 8, 8, W // 8, 8, C // 32, 32)
        #a_transformed_step2 = a_transformed_step1.transpose(
        #    0, 1, 3, 5, 2, 4, 6
        #)

        # Q: What does transpose ^^^ actually do above?  Why not just reshape immediately?
        # Does it have to do with numpy's 'reshape' when it increases the rank?

        #input_shape = [1,1,1,1,8,8,32]
        #output_shape = [1,8,8,32]

        ## Create the I/O tensors...
        #A_hexagon = tvm.nd.empty(input_shape, dtype, hexagon_session.device, 'global')
        #C_hexagon = tvm.nd.empty(output_shape, dtype, hexagon_session.device, 'global')

        #foo = tvm.nd.empty(input_shape, dtype, hexagon_session.device, 'global')


        ## Use a host-side tensor to provide the initial values for the
        ## primfunc call's input tensor...
        #A_host = np.ndarray(input_shape, dtype=dtype)

        #import random
        #for i0 in range(input_shape[0]):
        #    for i1 in range(input_shape[1]):
        #        for i2 in range(input_shape[2]):
        #            for i3 in range(input_shape[3]):
        #                for i4 in range(input_shape[4]):
        #                    for i5 in range(input_shape[5]):
        #                        for i6 in range(input_shape[6]):
        #                            A_host[i0,i1,i2,i3,i4,i5,i6] = random.randint(-128,127)

        a_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=input_shape_7d,
            axis_separators=[4],
            dtype="int8",
            mem_scope="global.vtcm",
        )

        c_hexagon = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=output_shape_4d,
            axis_separators=[],
            dtype="int8",
            mem_scope="global.vtcm",
        )

        a_hexagon.copyfrom(a_transformed)

        #a = tvm.nd.array(
        #    a_transformed,
        #    device=hexagon_session.device,
        #)
        #c = tvm.nd.array(c_np, device=hexagon_session.device)
        #mod(a, c)
        mod(a_hexagon, c_hexagon)

        tvm.testing.assert_allclose(ref_output, c_hexagon.numpy(), rtol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
