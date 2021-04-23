#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def transpose2(name : str, x, perm):
    import paddle as pdpd
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    out = pdpd.transpose(node_x, perm=perm)

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
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    transpose2("transpose2", data, perm=[1,0,2])

if __name__ == "__main__":
    main()     