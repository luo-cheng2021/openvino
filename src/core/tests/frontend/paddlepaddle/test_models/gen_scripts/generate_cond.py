import os
import sys

import numpy as np
import paddle

from save_model import exportModel

'''
test: 
'''
x = np.full(shape=[1], dtype='float32', fill_value=0.1)
y = np.full(shape=[1], dtype='float32', fill_value=0.23)
data = np.less(y,x)

@paddle.jit.to_static
def test_model(pred):
    # pred: A boolean tensor whose numel should be 1.
    def true_func():
        return paddle.full(shape=[3, 4], dtype='float32', # TODO: FAILED with different dtype
                        fill_value=1)

    def false_func():
        return paddle.full(shape=[1, 2], dtype='float32',
                        fill_value=3)

    return paddle.static.nn.cond(pred, true_func, false_func)

exportModel('conditional_block_const', test_model, [data], target_dir=sys.argv[1])


'''
more than one select_input
'''
@paddle.jit.to_static
def test_model_2outputs(pred):
    # pred: A boolean tensor whose numel should be 1.
    def true_func():
        return paddle.full(shape=[1, 2], dtype='float32',
                        fill_value=1), paddle.full(shape=[1, 3], dtype='float32', # TODO: FAILED with different dtype
                        fill_value=3)

    def false_func():
        return paddle.full(shape=[3, 4], dtype='float32',
                        fill_value=3), paddle.full(shape=[1, 4], dtype='float32',
                        fill_value=4)

    return paddle.static.nn.cond(pred, true_func, false_func)

exportModel('conditional_block_const_2outputs', test_model_2outputs, [data], target_dir=sys.argv[1])


'''
more than one select_input with 2 inputs
'''
@paddle.jit.to_static
def test_model_2inputs_2outputs(a, b):
    return paddle.static.nn.cond(a < b, lambda: (a, a * b), lambda: (b, a * b) )

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
exportModel('conditional_block_2inputs_2outputs', test_model_2inputs_2outputs, [a, b], target_dir=sys.argv[1])

'''
'''
@paddle.jit.to_static
def test_model2(a, b):
    c = a * b
    return paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
exportModel('conditional_block_2inputs', test_model2, [a, b], target_dir=sys.argv[1])


'''
'''
@paddle.jit.to_static
def test_model_dyn(a, b):
    c = a * b
    return a + c if a < b else b * b

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
exportModel('conditional_block_2inputs_dyn', test_model_dyn, [a, b], target_dir=sys.argv[1])


'''
more than one select_input
# looks there are bugs in paddle dyngraph to static... failed to generate 2 select_inputs.
'''
@paddle.jit.to_static
def test_model_dyn_2outputs(a, b):
    c = a * b
    return a, c  if a < b else b, c

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
exportModel('conditional_block_2inputs_dyn_2outputs', test_model_dyn_2outputs, [a, b], target_dir=sys.argv[1])
