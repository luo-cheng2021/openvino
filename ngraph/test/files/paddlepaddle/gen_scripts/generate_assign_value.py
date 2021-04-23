import numpy as np
from save_model import saveModel

def pdpd_assign_value(test_x):
    import paddle as pdpd
    pdpd.enable_static()
    node_x = pdpd.static.data(name='x', shape=test_x.shape, dtype='float32')
    const_value = pdpd.assign(test_x, output=None)
    add_node = pdpd.add(node_x, const_value);
    result = pdpd.static.nn.batch_norm(add_node, use_global_stats=True, epsilon=0)
    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())
    outs = exe.run(
        feed={'x': test_x},
        fetch_list=[result]
    )
    pdpd.static.save_inference_model("../models/paddle_assign_value",
                                     [node_x], [result], exe)
    saveModel("paddle_assign_value", exe, feedkeys=['x'], fetchlist=[result], inputs=[test_x], outputs=[outs[0]])

    print(outs[0])


def compare():
    x = np.ones([1, 1, 4, 4]).astype(np.float32)
    pdpd_result = pdpd_assign_value(x)


if __name__ == "__main__":
    compare()
