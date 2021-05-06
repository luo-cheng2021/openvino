import sys
import os
from generate_ir import ov_frontend_run
from openvino.inference_engine import IECore
import numpy as np
import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.core as core
import glob
import ngraph as ng
import collections
import logging
import pathlib
import save_model

# model params
models = {
    'test': {
        'model_pp': '/work/others/openvino/ngraph/test/files/paddlepaddle/models/conv2d_strides_no_padding/',
        'model_ov': '/work/others/openvino/ngraph/test/files/paddlepaddle/models/conv2d_strides_no_padding/conv2d_strides_no_padding.xml',
        'shapes': {'x': [1, 1, 7, 5]},
        'run': {
            'input': [
                {'x': np.random.rand(1, 1, 7, 5).astype('float32')},
            ],
            'fetch': 'all', #['x', 'conv2d_3.tmp_0'],
        },
        'valid': True
    },
    'ocr_det': {
        'model_pp': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/ocr_openvino_support/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel',
        'model_ov': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/ocr_openvino_support/ch_ppocr_mobile_v2.0_det_infer/inference.xml',
        'shapes': {'x': [1, 3, 640, 640]},
        'run': {
            'input': [
                {'x': np.random.rand(1, 3, 640, 640).astype('float32')},
                {'x': np.random.rand(1, 3, 640, 640).astype('float32')},
            ],
            'fetch':'all', #['x', 'conv2d_57.tmp_0', 'elementwise_add_0'],
        },
        'valid': True
    },
    'ocr_rec': {
        'model_pp': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/ocr_openvino_support/ch_ppocr_mobile_v2.0_rec_infer/inference.pdmodel',
        'model_ov': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/ocr_openvino_support/ch_ppocr_mobile_v2.0_rec_infer/inference.xml',
        'shapes': {'x': [1, 3, 32, 100]},
        'run': {
            'input': [
                {'x': np.random.rand(1, 3, 32, 100).astype('float32')},
                {'x': np.random.rand(1, 3, 32, 100).astype('float32')},
            ],
            'fetch': 'all', #[],
        },
        'valid': True
    },
    'ocrnet': {
        'model_pp': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino_support_seg/models/OCRNet/inference.pdmodel',
        'model_ov': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino_support_seg/models/OCRNet/inference.xml',
        'shapes': {'x': [1, 3, 1024, 512]},
        'run': {
            'input': [
                {'x': np.random.rand(1, 3, 1024, 512).astype('float32')},
            ],
            'fetch': 'all', #['conv2d_316.tmp_0'],
        },
        'valid': False
    },
    'yolov3': {
        'model_pp': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino-det/openvino/yolov3_darknet53_270e_coco/model.pdmodel',
        'model_ov': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino-det/openvino/yolov3_darknet53_270e_coco/model.xml',
        'shapes': {'scale_factor': [1, 2],
                'image': [1, 3, 608, 608],
                'im_shape': [1, 2]
            },
        'run': {
            'input': [
                {
                'scale_factor': np.array([1, 1]).astype('float32'),
                'image': np.random.rand(1, 3, 608, 608).astype('float32'),
                'im_shape': np.array([128, 128]).astype('float32'),
                },
            ],
            'fetch': 'all',
        },
        'valid': False
    },
    'ppyolo': {
        'model_pp': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino-det/openvino/ppyolo_r50vd_dcn_1x_coco/model.pdmodel',
        'model_ov': '/home/lc/paddle_models/20210427GUANZHONG-JIAJUN/openvino-det/openvino/ppyolo_r50vd_dcn_1x_coco/model.xml',
        'shapes': {'scale_factor': [1, 2],
                'image': [1, 3, 608, 608],
                'im_shape': [1, 2]
            },
        'run': {
            'input': [
                {
                'scale_factor': np.array([1, 1]).astype('float32'),
                'image': np.random.rand(1, 3, 608, 608).astype('float32'),
                'im_shape': np.array([128, 128]).astype('float32'),
                },
            ],
            'fetch': 'all',
        },
        'valid': False
    },
}

#models_dir = pathlib.Path(__file__).parents[1] / 'models'
###############################################
# log config section
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue = '\033[34m'
    green = '\033[92m'
    yellow = '\033[33m'
    red = '\033[91m'
    bold_red = '\033[91m'
    reset = "\x1b[0m"
    format = "%(asctime)s-%(levelname)s(%(lineno)d): %(message)s"
    datefmt = '%m-%d %H:%M:%S'

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
if logger.hasHandlers():
    logger.handlers = []
logger.addHandler(ch)
###############################################

def pp_run_one(model_path, input, fetch_list):
    def append_fetch_ops(program, fetch_target_names, fetch_holder_name='fetch'):
        """
        In this palce, we will add the fetch op
        """
        global_block = program.global_block()
        fetch_var = global_block.create_var(
            name=fetch_holder_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)
        #print("the len of fetch_target_names:%d" % (len(fetch_target_names)))
        for i, name in enumerate(fetch_target_names):

            global_block.append_op(
                type='fetch',
                inputs={'X': [name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            
    def insert_fetch(program, fetchs, fetch_holder_name="fetch"):
        global_block = program.global_block()
        need_to_remove_op_index = list()
        for i, op in enumerate(global_block.ops):
            if op.type == 'fetch':
                need_to_remove_op_index.append(i)
        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)
        program.desc.flush()
        append_fetch_ops(program, fetchs, fetch_holder_name)

    def test():
        # load model
        exe = fluid.Executor(fluid.CPUPlace())
        dirs, name  = os.path.split(model_path)
        if len(name) > 0:
            [prog, feed, fetchs] = fluid.io.load_inference_model(
                                dirs, 
                                exe, name, name.split('.')[0] + '.pdiparams')
        else:
            [prog, feed, fetchs] = fluid.io.load_inference_model(
                                dirs, 
                                exe)

        all_fetch_list = [fetch.name for fetch in fetchs] + fetch_list
        insert_fetch(prog, all_fetch_list)
        fetch_tensor = [prog.global_block().var(fetch_name) for fetch_name in all_fetch_list]

        results = []
        for val in input:
            result = exe.run(prog, feed=val, fetch_list=fetch_tensor, return_numpy=True)
            result_dict = {}
            for i, fetch in enumerate(all_fetch_list):
                result_dict[fetch] = result[i]
            results.append(result_dict)
        return results

    def test_jit():
        paddle.disable_static()
        # jit.load will change the original name in network
        model = paddle.jit.load('.'.join(model_path.split('.')[0:-1]))
        model.eval()
        results = []
        for val in input:
            result = model(*list(val.values()))
            result_dict = {}
            result_dict[result.name] = result.numpy()
            results.append(result_dict)
        paddle.enable_static()

        return results

    return test()

def ov_run_one(model_path, input, fetch_list):
    def save_graph_as_pic(function):
        from ngraph.impl.passes import Manager
        pass_manager = Manager()
        # should modify ngraph/python/src/pyngraph/passes/manager.cpp register 'VisualizeTree'
        pass_manager.register_pass('VisualizeTree')
        pass_manager.run_passes(function)

    ie = IECore()
    network = ie.read_network(model=model_path)
    device = 'CPU'
    # insert fetch list
    if len(fetch_list):
        #device = 'TEMPLATE'
        f = ng.function_from_cnn(network)
        ops = f.get_ordered_ops()
        results_op = []
        fetch_list_remain = fetch_list.copy()
        for node in ops:
            if node.get_friendly_name() in fetch_list:
                result_op = ng.result(node)
                results_op.append(result_op)
                fetch_list_remain.remove(node.get_friendly_name())
        if len(fetch_list_remain):
            first_batch = True
            # remove many batch_norm_xxx, it's so boring
            fixed_fetch_list_remain = []
            for i in fetch_list_remain:
                if i.startswith('batch_norm_'):
                    if first_batch:
                        first_batch = False
                        fixed_fetch_list_remain.append(i + '...')
                else:
                    fixed_fetch_list_remain.append(i)
            logger.warning(f'cannot insert fetch list: {fixed_fetch_list_remain}')
        updated_f = ng.Function(f.get_results() + results_op, f.get_parameters(), f.get_friendly_name())
        save_graph_as_pic(updated_f)
        network = ng.function_to_cnn(updated_f)

    executable_network = ie.load_network(network, device)
    net_inputs = network.input_info
    outputs = []
    for i in input:
        for name in i.keys():
            if name not in net_inputs:
                logger.warning(f'{name} not in inputs')
        output = executable_network.infer(i)
        # order the result
        order_result = collections.OrderedDict()
        for f in fetch_list:
            if f in output:
                order_result[f] = output[f]
        outputs.append(order_result)
    return outputs

def get_ops_info_by_output_name(model_path, output_name, attrib_name):
    # load model
    exe = fluid.Executor(fluid.CPUPlace())
    dirs, name  = os.path.split(model_path)
    if len(name) > 0:
        [prog, feed, fetchs] = fluid.io.load_inference_model(
                            dirs, 
                            exe, name, name.split('.')[0] + '.pdiparams')
    else:
        [prog, feed, fetchs] = fluid.io.load_inference_model(
                            dirs, 
                            exe)

    for i, op in enumerate(prog.blocks[0].ops):
        output_names = op.output_arg_names
        if output_name in output_names:
            return op.input_arg_names, op.attr(attrib_name) if len(attrib_name) else None, op

    return None

def handle_known_error(model_path, output_name, v_ov, v_pp, results_ov, results_pp):
    if output_name.startswith('argmax_'):
        # openvino argmax uses topk to simulate and its doc says:
        #     'If there are several elements with the same value then their output order is not determined.'
        #      https://docs.openvinotoolkit.org/latest/openvino_docs_ops_sort_TopK_3.html
        # we should output some useful information to check it
        input_names, axis, _ = get_ops_info_by_output_name(model_path, output_name, 'axis')
        assert(len(input_names) == 1)
        assert(input_names[0] in results_ov and input_names[0] in results_pp)
        result_ov = results_ov[input_names[0]]
        result_pp = results_pp[input_names[0]]
        non_eq = np.where((v_ov - v_pp) != 0)
        idxs = [tuple(x) for x in zip(*non_eq)]
        failed = False
        for idx in idxs:
            # the different argmax value in ov/pp
            val_ov_idx = v_ov[idx]
            val_pp_idx = v_pp[idx]
            input_ov_idx = list(idx)
            input_ov_idx.insert(axis, val_ov_idx)
            input_ov_idx = tuple(input_ov_idx)
            input_pp_idx = list(idx)
            input_pp_idx.insert(axis, val_pp_idx)
            input_pp_idx = tuple(input_pp_idx)
            # the value in argmax input
            val_ov = result_ov[input_ov_idx]
            val_pp = result_pp[input_pp_idx]
            if np.abs(val_ov - val_pp) < 0.1:
                logger.info(f'argmax: ok ov[{input_ov_idx}]={val_ov}, pp[{input_pp_idx}]={val_pp}; ov[{input_pp_idx}]={result_ov[input_pp_idx]}, pp[{input_ov_idx}]={result_pp[input_ov_idx]}')
            else:
                failed = True
                logger.error(f'argmax: error ov[{input_ov_idx}]={val_ov}, pp[{input_pp_idx}]={val_pp}; ov[{input_pp_idx}]={result_ov[input_pp_idx]}, pp[{input_ov_idx}]={result_pp[input_ov_idx]}')
                break

        return not failed

    return False

def try_gen_model_by_output_name(input_parameter, model_path, output_name, results_pp):
    # get op's info
    input_arg_names, _, op = get_ops_info_by_output_name(model_path, output_name, '')
    output_arg_names = op.output_arg_names
    
    # gen model
    should_save = False
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    if op.type == 'conv2d':
        filter_name = op.input('Filter')[0]
        data_name = op.input('Input')[0]
        with paddle.static.program_guard(main_program, startup_program):
            filter_numpy = pp_run_one(model_path, [input_parameter], [filter_name])
            kernel = filter_numpy[0][filter_name]
            data = paddle.static.data(name='x', shape=results_pp[data_name].shape, dtype='float32')
            weight_attr = paddle.ParamAttr(name="conv2d_weight", initializer=paddle.nn.initializer.Assign(kernel))
            conv2d = paddle.static.nn.conv2d(input=data, num_filters=kernel.shape[0], filter_size=kernel.shape[2:4],
                                        padding=op.attr('paddings'), param_attr=weight_attr, 
                                        dilation=op.attr('dilations'), stride=op.attr('strides'), groups=op.attr('groups'), 
                                        use_cudnn=op.attr('use_cudnn'))
            cpu = paddle.static.cpu_places(1)
            exe = paddle.static.Executor(cpu[0])
            exe.run(startup_program)
            outs = exe.run(
                feed={'x': results_pp[data_name]},
                fetch_list=conv2d,
                program=main_program)
            # check new model output is same with the orginal op's result
            s = np.sum(np.abs(outs[0] - results_pp[output_name]))
            if s > 0.0001:
                logger.error(f'generate for {output_name} failed, error is too much {s}')
            save_model.saveModel('_' + output_name, exe, ['x'], conv2d, [results_pp[data_name]], [outs[0]])
            logger.debug(f'generated model _{output_name} ok.')
    else:
        logger.warning(f'can not support save model {output_name}')

def cmp_results(models, results, model_name):
    result_pp = results['pp'][model_name]
    for i, r_ov in enumerate(results['ov'][model_name]):
        r_pp = result_pp[i]
        failed = False
        for name, value in r_ov.items():
            value_pp = r_pp[name]
            v1 = np.array(value)
            v2 = np.array(value_pp)
            s = np.sum(np.abs(v1 - v2)) / v2.size
            m = np.max(np.abs(v1 - v2))
            if s > 0.0001 or m > 0.001:
                logger.error(f'result[{i}]["{name}"] is not same: avg error:{s} max error: {m}')
                if handle_known_error(models[model_name]['model_pp'], name, v1, v2, r_ov, r_pp):
                    logger.warning("maybe it's not a bug")
                else:
                    failed = True
                try_gen_model_by_output_name(models[model_name]['run']['input'][i], models[model_name]['model_pp'], name, r_pp)
            #else:
                #print(f'result[{i}]["{name}"] is ok: avg error:{s}, max error: {m}')
        if not failed:
            logger.info(f'result[{i}] is ok')

def get_ops_output_names(model_path):
    # load model
    exe = fluid.Executor(fluid.CPUPlace())
    dirs, name  = os.path.split(model_path)
    if len(name) > 0:
        [prog, feed, fetchs] = fluid.io.load_inference_model(
                            dirs, 
                            exe, name, name.split('.')[0] + '.pdiparams')
    else:
        [prog, feed, fetchs] = fluid.io.load_inference_model(
                            dirs, 
                            exe)

    ops_names_in = []
    ops_names_out = []
    for i, op in enumerate(prog.blocks[0].ops):
        ops_names_in += op.input_arg_names
        output_names = op.output_names
        if 'Out' in output_names:
            ops_names_out += op.output('Out')
        elif 'Output' in output_names:
            ops_names_out += op.output('Output')
        elif 'Y' in output_names:
            ops_names_out += op.output('Y')
        else:
            logger.error(f'Unknown out name: {output_names}, {op.output_arg_names}')
    ops_names_out = [x for x in ops_names_out if x != 'fetch']
    ops_names = ops_names_out

    return ops_names

def prepare_configs(models):
    for m in models.keys():
        if models[m]['valid'] == False:
            continue

        if models[m]['run']['fetch'] == 'all':
            ops_name_list = get_ops_output_names(models[m]['model_pp'])
            models[m]['run']['fetch'] = ops_name_list

def main():
    global models
    # pd exec and its output
    results = {
        'pp': {},
        'ov': {}
    }

    prepare_configs(models)

    for model_name, info in models.items():
        if not info['valid']:
            logger.debug(f'skip model {model_name}')
            continue

        # convert pd model => ov model
        logger.debug(f'convert pd model {model_name} to ov ...')
        def _get_fixed_ppmodel_info(info):
            # if using save_inference_model we should pass the dir only to load_inference_model, not the fullpath
            info_copy = info.copy()
            dirs, name = os.path.split(info_copy['model_pp'])
            if len(name) == 0:
                model_path = glob.glob(dirs + '/*.pdmodel')
                if len(model_path) == 0:
                    logger.error(f'{dirs} has no pdmodel file')
                    raise Exception("no pdmodel file found")
                info_copy['model_pp'] = model_path[0]
            return info_copy
        info_copy = _get_fixed_ppmodel_info(info)
        ov_frontend_run(info_copy['model_pp'], info_copy)

        # run pp model
        logger.debug(f'running pd model {model_name} ...')
        results['pp'][model_name] = pp_run_one(info['model_pp'], info['run']['input'], info['run']['fetch'])

        # run ov model
        logger.debug(f'running ov model {model_name} ...')
        results['ov'][model_name] = ov_run_one(info['model_ov'], info['run']['input'], info['run']['fetch'])

        # cmp pp <=> ov result
        logger.debug(f'comparing {model_name} ov and pp result ...')
        cmp_results(models, results, model_name)

    logger.debug('all done.')        

if __name__ == "__main__":
    main()
