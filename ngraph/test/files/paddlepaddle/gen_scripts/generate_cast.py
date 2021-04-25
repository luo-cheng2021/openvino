#
# cast paddle model generator
#
import numpy as np
from save_model import saveModel

def cast(name : str, x, in_dtype, out_dtype):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        out = pdpd.cast(node_x, out_dtype)

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
    # TODO: more type
    in_dtype = 'float32'
    out_dtype = 'float32'
    data = np.array( [ [1.1, 2.1, 1.0], [3.2, 4.7, 5.6] ], dtype = in_dtype )
    cast("cast_test1", data, in_dtype, out_dtype)

#    in_dtype = 'float32'
#    out_dtype = 'uint8'
#    data = np.array( [ [1.1, 2.1, 1], [3.2, 4, 5] ], dtype = in_dtype )
#    cast("cast_test2", data, in_dtype, out_dtype)

if __name__ == "__main__":
    main()
