#
# pow paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def pow(name : str, x, y):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.pow(node_x, y, name = 'pow')

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]

def pow_tensor(name : str, x, y):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        node_f = pdpd.assign(y, output=None)
        out = pdpd.fluid.layers.pow(node_x, node_f, name = 'pow')

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
    x = np.array([0, 1, 2, -10]).astype(data_type)

    pow("pow", x, 2.0)
    pow_tensor("pow_tensor", x, np.array(2.0).astype(data_type))

if __name__ == "__main__":
    main()