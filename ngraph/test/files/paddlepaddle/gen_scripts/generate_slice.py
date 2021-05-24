#
# slice paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def slice(name : str, x, axes : list, start : list, end : list):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.slice(node_x, axes = axes, starts = start, ends = end)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]

def slice_tensor(name : str, x, axes : list, start : list, end : list, use_tensor_in_list : bool):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        if use_tensor_in_list:
            start[0] = pdpd.assign(np.array((start[0],)).astype('int32'))
            end[0] = pdpd.assign(np.array((end[0],)).astype('int32'))
            out = pdpd.fluid.layers.slice(node_x, axes=axes, starts=start, ends=end)
        else:
            out = pdpd.fluid.layers.slice(node_x, axes = axes, \
                starts = pdpd.assign(np.array(start).astype('int32')), \
                ends = pdpd.assign(np.array(end).astype('int32')))

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]

def main():
    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(4, 3, 5).astype(data_type)
    slice("slice", x, axes=[1, 2], start=(0, 1), end=(-1, 3))

    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(2, 30).astype(data_type)
    slice("slice_1d", x, axes=[0], start=[0], end=[1])

    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(4, 3, 5).astype(data_type)
    slice_tensor("slice_tensor", x, axes=[1, 2], start=(0, 1), end=(-1, 3), use_tensor_in_list=False)
    slice_tensor("slice_tensor_list", x, axes=[1, 2], start=[0, 1], end=[-1, 3], use_tensor_in_list=True)

if __name__ == "__main__":
    main()