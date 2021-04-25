#
# elementwise paddle model generator
#
import numpy as np
from save_model import saveModel

import numpy as np
from save_model import saveModel

def elementwise_add(name : str, x, y, in_dtype):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = pdpd.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = pdpd.fluid.layers.nn.elementwise_add(node_x, node_y)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]])

    return outs[0]

def elementwise_sub(name : str, x, y, in_dtype):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = pdpd.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = pdpd.fluid.layers.nn.elementwise_sub(node_x, node_y)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]])

    return outs[0]

def elementwise_div(name : str, x, y, in_dtype):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = pdpd.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = pdpd.fluid.layers.nn.elementwise_div(node_x, node_y)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]])

    return outs[0]

def elementwise_mul(name : str, x, y, in_dtype):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = pdpd.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = pdpd.fluid.layers.nn.elementwise_mul(node_x, node_y)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]])

    return outs[0]

def main():
    # TODO: more type
    in_dtype = 'float32'
    data_x = np.array([2, 3, 4]).astype(in_dtype)
    data_y = np.array([1, 5, 2]).astype(in_dtype)

    elementwise_add("elementwise_add1", data_x, data_y, in_dtype)
    elementwise_sub("elementwise_sub1", data_x, data_y, in_dtype)
    elementwise_div("elementwise_div1", data_x, data_y, in_dtype)
    elementwise_mul("elementwise_mul1", data_x, data_y, in_dtype)

if __name__ == "__main__":
    main()
