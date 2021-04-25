#
# log paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def log(name : str, x):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.log(node_x, name = 'log')

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

    log("log", x)

if __name__ == "__main__":
    main()