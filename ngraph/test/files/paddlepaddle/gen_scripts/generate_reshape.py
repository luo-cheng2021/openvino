#
# reshape paddle model generator
#
import numpy as np
from save_model import saveModel

data_type = 'float32'

def reshape(name : str, x, out_shape):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = pdpd.fluid.layers.reshape(x=node_x, shape=out_shape)

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
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)
    out_shape = [1, 1, 2, 8]
    reshape("reshape", data, out_shape)
 


if __name__ == "__main__":
    main()     