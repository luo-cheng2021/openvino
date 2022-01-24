import os
import sys

import numpy as np
import paddle

from save_model import exportModel

""""""""""""""""""""""""""""""""""""""""""""""""""""""
# tensorarray case: conditional_block + slice[0]
""""""""""""""""""""""""""""""""""""""""""""""""""""""
def test_conditional_block_slice0():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]    
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_slice0', [a])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice0_dyn', [a], [a_shape])

def test_conditional_block_slice0_scaler():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            return exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor(10.0)
    print(test('conditional_block_slice0_scaler', [a]))
    a_shape = a.shape
    a_shape[0] = -1
    print(test('conditional_block_slice0_scaler_dyn', [a], [a_shape]))

def test_conditional_block_slice1():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return rpn_rois_list[1]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])    
    test('conditional_block_slice1', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice1_dyn', [a, b], [a_shape, b.shape])

def test_conditional_block_slice0_slice1():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return rpn_rois_list[0], rpn_rois_list[1]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])                
    test('conditional_block_slice0_slice1', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice0_slice1_dyn', [a, b], [a_shape, b.shape])

def test_conditional_block_concat():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return paddle.concat(rpn_rois_list)

            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])
    test('conditional_block_concat', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_concat_dyn', [a, b], [a_shape, b.shape])

if __name__ == "__main__":
    test_conditional_block_slice0()
    test_conditional_block_slice1()
    test_conditional_block_slice0_slice1()
    test_conditional_block_slice0_scaler()
    test_conditional_block_concat()
