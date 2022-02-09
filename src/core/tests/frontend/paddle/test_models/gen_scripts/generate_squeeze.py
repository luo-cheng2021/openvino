# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# squeeze paddle model generator
#
import numpy as np
from save_model import saveModel, exportModel
import paddle
import sys

data_type = 'float32'

def squeeze(name : str, x, axes : list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.fluid.layers.squeeze(node_x, axes=axes, name='squeeze')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def slice_dyn(test_shape=[2,8,10,10]):
    paddle.disable_static()

    data = paddle.rand(shape=test_shape, dtype='float32')

    '''
    slice w/ decrease_axis
    '''
    @paddle.jit.to_static
    def test_slice_decrease_axis(x):
        return x[0, 1:3, :, 5]
    exportModel('slice_decrease_axis', test_slice_decrease_axis, [data], target_dir=sys.argv[1]) # output shape (2, 10)

    '''
    slice w/o decrease_axis
    '''
    @paddle.jit.to_static
    def test_slice(x):
        return paddle.slice(x, axes=[0,1,3], starts=[0,1,5], ends=[1,3,6])
    # exportModel('slice_dyn', test_slice, [data], target_dir=sys.argv[1]) # output shape (1, 2, 10, 1)  # disable it by default as this kind of test model already there. It's for comparsion only.

    '''
    slice w/ decrease_axis of all dims
    '''
    @paddle.jit.to_static
    def test_slice_decrease_axis_all(x):
        return x[0, 0, 0, 0]
    exportModel('slice_decrease_axis_all', test_slice_decrease_axis_all, [data], target_dir=sys.argv[1]) # output shape (1,)

    '''
    slice w/o decrease_axis of all dims
    '''
    @paddle.jit.to_static
    def test_slice_alldim(x):
        return paddle.slice(x, axes=[0,1,2,3], starts=[0,0,0,0], ends=[1,1,1,1])
    # exportModel('slice_alldim', test_slice_alldim, [data], target_dir=sys.argv[1]) # output shape (1, 1, 1, 1) # disable it by default as this kind of test model already there. It's for comparsion only.

'''
a test case simulating the last reshape2 of ocrnet which accepts slice (with decrease_axes in all dims) as its parents.
'''
def slice_reshape(B=1, C=256, H=16, W=32):
    paddle.disable_static()

    data = paddle.rand(shape=[B, C, H*W], dtype='float32')

    @paddle.jit.to_static
    def test_model(x):
        x2 = paddle.assign([-1, -1, 16, 32]).astype('int32')
        node_reshape = paddle.reshape(x, [0, 256, x2[2], x2[3]])
        return node_reshape
    exportModel('slice_reshape', test_model, [data], target_dir=sys.argv[1])

def main():
    data = np.random.rand(1, 3, 1, 4).astype(data_type)

    squeeze("squeeze", data, [0, -2])
    squeeze("squeeze_null_axes", data, [])

if __name__ == "__main__":
    main()
    slice_dyn()
    slice_reshape()