import os
import time

from typing import Optional, Tuple
import numpy as np

from openvino.runtime import Core, Model, Tensor, PartialShape, serialize, Type, Shape
from openvino.runtime.passes import Manager
from openvino.runtime.passes import VisualizeTree

def test(path):
    input_ids_np = np.load('input_ids.npy', allow_pickle=True)
    core = Core()
    net = core.read_model(model=path)
    batch = input_ids_np.shape[0]
    config = {'PERFORMANCE_HINT': '',
        'NUM_STREAMS': '1',
        'INFERENCE_PRECISION_HINT': 'f32',
        'CPU_RUNTIME_CACHE_CAPACITY': '5000000',
        'AFFINITY': 'CORE',
        #'PERFORMANCE_HINT_NUM_REQUESTS': '2'
        #'ENFORCE_BF16': 'YES'
        'INFERENCE_NUM_THREADS': '1' #'64'
        }
    seq_len = input_ids_np.shape[1]
    net.reshape({'input_ids': [batch, seq_len], #[2, -1], # [-1, -1],
                        })
    exec_net1 = core.compile_model(net, 'CPU', config)
    req1 = exec_net1.create_infer_request()
    stat = {
        'init': 0,
        'infer_1x300': 0,
        'infer_1x1': 0,
        'post': 0,
        'times': 0
    }

    #input_ids_np = (np.ones([1, 1]) * 1000).astype(np.int64)
    past_key_num = np.array([0,], dtype=np.int64)
    inputs1 = {
        0: Tensor(input_ids_np),
        1: Tensor(past_key_num),
    }

    beg = time.time()
    req1.set_tensors(inputs1)
    req1.infer()
    if inputs1[0].shape[1] == 300:
        stat['infer_1x300'] += time.time() - beg
    else:
        stat['infer_1x1'] += time.time() - beg
    stat['times'] += 1
    model = exec_net1.get_runtime_model()
    #serialize(model, 'exec1.xml', 'exec1.bin')
    print(req1.outputs[0].data)

    input_ids = [33534, 42621]
    offset = 306
    #print(stat)

def test1x1(path):
    input_ids_np = np.array([[33534], [42621]], dtype=np.int64)
    core = Core()
    net = core.read_model(model=path)
    batch = input_ids_np.shape[0]
    config = {'PERFORMANCE_HINT': '',
        'NUM_STREAMS': '1',
        'INFERENCE_PRECISION_HINT': 'f32',
        'CPU_RUNTIME_CACHE_CAPACITY': '5000000',
        'AFFINITY': 'CORE',
        #'PERFORMANCE_HINT_NUM_REQUESTS': '2'
        #'ENFORCE_BF16': 'YES'
        'INFERENCE_NUM_THREADS': '1' #'64'
        }
    seq_len = input_ids_np.shape[1]
    net.reshape({'input_ids': [batch, seq_len], #[2, -1], # [-1, -1],
                        })
    exec_net1 = core.compile_model(net, 'CPU', config)
    req1 = exec_net1.create_infer_request()
    stat = {
        'init': 0,
        'infer_1x300': 0,
        'infer_1x1': 0,
        'post': 0,
        'times': 0
    }

    #input_ids_np = (np.ones([1, 1]) * 1000).astype(np.int64)
    past_key_num = np.array([305,], dtype=np.int64)
    inputs1 = {
        0: Tensor(input_ids_np),
        1: Tensor(past_key_num),
    }

    beg = time.time()
    req1.set_tensors(inputs1)
    req1.infer()
    if inputs1[0].shape[1] == 300:
        stat['infer_1x300'] += time.time() - beg
    else:
        stat['infer_1x1'] += time.time() - beg
    stat['times'] += 1
    model = exec_net1.get_runtime_model()
    #serialize(model, 'exec1.xml', 'exec1.bin')
    print(req1.outputs[0].data)

test('./hacked/gpt_neox.xml')
#test1x1('./hacked/gpt_neox.xml')