#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def batch_norm(name : str, x, scale, bias, mean, var):
    import paddle as pdpd
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    scale_attr = pdpd.ParamAttr(name="scale", initializer=pdpd.nn.initializer.Assign(scale))
    bias_attr = pdpd.ParamAttr(name="bias", initializer=pdpd.nn.initializer.Assign(bias))

    out = pdpd.static.nn.batch_norm(node_x, epsilon=1e-5,
                                   param_attr=scale_attr,
                                   bias_attr=bias_attr,
                                   moving_mean_name="bn_mean",
                                   moving_variance_name="bn_variance",
                                   use_global_stats=True)

    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())
    pdpd.static.global_scope().var("bn_mean").get_tensor().set(mean, pdpd.CPUPlace())
    pdpd.static.global_scope().var("bn_variance").get_tensor().set(var, pdpd.CPUPlace())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])             

    saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]


def main():
    import paddle as pdpd
    data = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)

    scale = np.array([1.0, 1.5]).astype(np.float32)
    bias = np.array([0, 1]).astype(np.float32)
    mean = np.array([0, 3]).astype(np.float32)
    var = np.array([1, 1.5]).astype(np.float32)

    batch_norm("batch_norm", data, scale, bias, mean, var)

if __name__ == "__main__":
    main()     