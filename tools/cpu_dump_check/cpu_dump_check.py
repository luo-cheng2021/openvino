#!/usr/bin/python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core, Model, Tensor, PartialShape, Type
from openvino.runtime import opset8 as opset
from openvino.runtime.op import Constant, Parameter, tensor_iterator
from openvino.runtime.passes import Manager
from openvino.runtime.utils.types import get_dtype
import openvino as ov
import numpy as np
import sys
import os, errno
import struct
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

def mkdirp(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def fill_tensors_with_random(input):
    dtype = get_dtype(input.get_element_type())
    rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    shape = input.get_shape()
    a = rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)
    return Tensor(a)

class IEB:
    def __init__(self, ieb_file) -> None:
        with open(ieb_file,"rb") as f:
            data = f.read() # bytes
            header = struct.unpack_from("@4sHBB7IB3BLLLL", data, offset=0)
            # print(header, len(header))
            (self.magic, self.ver, self.precision, self.ndims,
            self.dims0, self.dims1, self.dims2, self.dims3, self.dims4, self.dims5, self.dims6,
            self.scaling_axis,
            self.reserved0, self.reserved1, self.reserved2,
            self.data_offset, self.data_size, self.scaling_data_offset, self.scaling_data_size) = header
            precision_table = {
                10:(np.float32, 4),
                40:(np.uint8, 1),
                50:(np.int8, 1),
                70:(np.int32, 4),
                74:(np.uint32, 4),
                72:(np.int64, 8),
                73:(np.uint64, 8)
            }
            (dtype, type_size, ) = precision_table[self.precision]
            count = self.data_size//type_size
            
            # recover the data as numpy array
            self.dims = np.array([self.dims0, self.dims1, self.dims2, self.dims3, self.dims4, self.dims5, self.dims6])
            self.dims = self.dims[0:self.ndims]
            self.value = np.frombuffer(data, dtype = dtype, count=count, offset=self.data_offset)
            self.value = np.reshape(self.value, self.dims)

            # self.values = struct.unpack_from(f"@{count}{stype}", data, offset=self.data_offset)
            # print(self.values.shape, self.values.dtype)
        pass

class DumpIndex:
    def __init__(self, args) -> None:
        (self.ExecIndex, self.Name, self.OriginalLayers, self.tag, self.itag, self.ieb_file) = args


def dump_tensors(core, model, dump_dir = "./cpu_dump", device_target="CPU"):
    os.environ["OV_CPU_BLOB_DUMP_DIR"] = dump_dir
    os.environ["OV_CPU_BLOB_DUMP_FORMAT"] = "BIN"
    os.environ["OV_CPU_BLOB_DUMP_NODE_PORTS"] = "OUT"
    mkdirp(dump_dir)

    device_config = {"PERF_COUNT": "NO",
                "AFFINITY": "CORE",
                "PERFORMANCE_HINT_NUM_REQUESTS":0,
                "PERFORMANCE_HINT":"",
                "NUM_STREAMS":1,
                "INFERENCE_NUM_THREADS":1}

    print("compiling model with {}".format(device_config))
    exec_net = core.compile_model(model, device_target, device_config)
    req = exec_net.create_infer_request()

    print("fill input with random data:")
    inputs={}
    for i in exec_net.inputs:
        inputs[i] = fill_tensors_with_random(i)
        print(f"  {i}")

    print("infer with dump..")
    req.infer(inputs)


def visualize_diff_abs(diff_abs):
    vis_abs = diff_abs
    cur_shape = diff_abs.shape
    if len(vis_abs.shape) > 3:
        vis_abs = vis_abs.reshape(-1,cur_shape[-2],cur_shape[-1])
    
    fig, ax = plt.subplots()
    im = ax.imshow(vis_abs[0,:,:])

    cur_channel = 0
    def update_channel(val):
        nonlocal cur_channel
        val = int(val)
        cur_channel = val
        diff_img = vis_abs[val,:,:]
        max_diff = np.amax(diff_img)
        ax.set_title(" channel:{}  shape:{}  Max diff: {:.8f}".format(
                        val, diff_img.shape, np.amax(diff_img)))
        # normalize intensity
        im.set_data(diff_img * 255 / max_diff)
        fig.canvas.draw_idle()

    update_channel(0)

    ax_ch_slider = plt.axes([0.1, 0.25, 0.0225, 0.63])
    ch_slider = Slider(
        ax=ax_ch_slider,
        label="Channels",
        valmin=0,
        valmax=vis_abs.shape[0],
        valinit=0,
        valstep=1,
        orientation="vertical"
    )

    ch_slider.on_changed(update_channel)

    def on_press(event):
        # print('press', event.key, 'cur_channel', cur_channel)
        sys.stdout.flush()
        if event.key == 'escape':
            print("escape key detected, exit.")
            sys.exit(1)
        if event.key == 'up':
            for c in range(cur_channel+1, vis_abs.shape[0]):
                diff_img = vis_abs[c,:,:]
                if np.amax(diff_img) > 1e-8:
                    ch_slider.set_val(c)
                    break
        if event.key == 'down':
            for c in range(cur_channel-1, -1, -1):
                diff_img = vis_abs[c,:,:]
                if np.amax(diff_img) > 1e-8:
                    ch_slider.set_val(c)
                    break
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

def compare_dumps(model, atol, visualize, dump_dir1, dump_dir2):

    output_tensors = []
    for out in model.outputs:
        for oname in out.get_names():
            output_tensors.append(oname.split(":")[0])

    def is_output(name):
        for tag in output_tensors:
            if tag in name:
                return True
        return False

    def get_sorted_ied_list(dir):
        iebs = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".ieb"):
                k = file_name.find("_")
                id = int(file_name[1:k])
                name = file_name[k:]
                iebs.append((id, name, file_name))
        return sorted(iebs, key=lambda item:item[0])

    ieb_list1 = get_sorted_ied_list(dump_dir1)
    ieb_list2 = get_sorted_ied_list(dump_dir2)

    def get_match_ieb_file2(f1):
        for f2 in ieb_list2:
            if f1[1] == f2[1]:
                return f2
        return None

    MAX_atol = {}
    for f1 in ieb_list1:
        f2 = get_match_ieb_file2(f1)
        if not f2:
            continue
        
        ieb_file1 = f1[-1]
        ieb_file2 = f2[-1]
        # compare 
        ieb1 = IEB(os.path.join(dump_dir1, ieb_file1))
        ieb2 = IEB(os.path.join(dump_dir2, ieb_file2))

        if "Input_Constant" in ieb_file1 and "Input_Constant" in ieb_file2:
            print("Skipped Input_Constant {ieb_file1} vs {ieb_file2}")
            continue

        if not np.allclose(ieb1.value, ieb2.value, atol=atol):
            diff_abs = np.abs(ieb1.value - ieb2.value)
            atol_max = np.amax(diff_abs)

            if ieb1.value.dtype in MAX_atol:
                if MAX_atol[ieb1.value.dtype] < atol_max:
                    MAX_atol[ieb1.value.dtype] = atol_max
            else:
                MAX_atol[ieb1.value.dtype] = 0

            prefixERR = Colors.RED
            if is_output(f1[-1]):
                prefixERR += Colors.UNDERLINE
            print("{}[  FAILED ]: {} {} {}".format(prefixERR, f1[-1], f2[-1], Colors.END))
            info  = ""
            if (np.prod(diff_abs.shape) < 8):
                info = "{} vs {}".format(ieb1.value.reshape(-1), ieb2.value.reshape(-1))
            
            print("    {} {}    ({:.2e} ~ {:.2e})   @ mean:{:.2e} std:{:.2e}  detail: {}".format(
                    diff_abs.shape, diff_abs.dtype,
                    np.amin(diff_abs), np.amax(diff_abs), np.mean(diff_abs), np.std(diff_abs), info))

            if (visualize):
                visualize_diff_abs(diff_abs)
        else:
            #print("{}[  OK     ]: {} {} {}".format(prefixOK, f1[-1], f2[-1], Colors.END))
            pass

    print("============================================")
    if (len(MAX_atol) == 0):
        print("Pass")
    else:
        for prec in MAX_atol:
            print("Max atol {} : {}".format(prec, MAX_atol[prec]))

def main():
    parser = argparse.ArgumentParser("cpu_cross_check")
    parser.add_argument("-m", type=str, default="", required=True, help="Model file path")
    parser.add_argument("-atol", type=float, default=1e-8, help="absolute error")
    parser.add_argument("-v", action="store_true", help="visualize error")
    parser.add_argument("dumps", type=str, default="", nargs="+", help="dump folders")
    args = parser.parse_args()

    print(f"Read model {args.m}...")
    core = Core()
    model = core.read_model(args.m)

    if len(args.dumps) == 1:
        dump_tensors(core, model, args.dumps[0])
    else:
        assert(len(args.dumps) == 2)
        compare_dumps(model, args.atol, args.v, args.dumps[0], args.dumps[1])


if __name__ == "__main__":
    main()






commented_code = '''
    dump_dir = "./cpu_dump"
    if os.path.isdir(dump_dir):
        print(f"using dump result from {dump_dir}, (re)move this folder if re-dump is required ...")
    else:
        dump_tensors(core, model, )

    index_file = os.path.join(dump_dir, "index.txt")
    print(f"Checking dumpped tensor index in {index_file}...")
    dump_tensors = []
    with open(index_file) as f:
        for l in f.readlines():
            l = l.rstrip("\n")
            di = DumpIndex(l.split(";"))
            dump_tensors.append(di)
            #ieb = IEB(di.ieb_file)
            #print(ieb.value)
    
    if (len(dump_tensors) == 0):
        print("No valid tensor was found!")
        return
    
    print("{} tensors was found!".format(len(dump_tensors)))
    # find node name in ngraph model and add them to result
    '''