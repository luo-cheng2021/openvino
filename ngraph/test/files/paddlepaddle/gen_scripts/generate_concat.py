import numpy as np
from save_model import saveModel

def pdpd_concat(name, test_x, test_y, axes):
    import paddle as pdpd
    pdpd.enable_static()
    node_x = pdpd.static.data(name='x', shape=test_x.shape, dtype='float32')
    node_y = pdpd.static.data(name='y', shape=test_y.shape, dtype='float32')
    node_concat = pdpd.concat([node_x, node_y], axis=axes)
    result = pdpd.static.nn.batch_norm(node_concat, use_global_stats=True, epsilon=0)
    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())
    outs = exe.run(
        feed={'x': test_x, "y":test_y},
        fetch_list=[result]
    )
    saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[result], inputs=[test_x, test_y], outputs=[outs[0]])
    print(outs[0])


if __name__ == "__main__":
    x = np.ones([1, 1, 4, 4]).astype(np.float32)
    y = np.zeros([1, 1, 4, 4]).astype(np.float32)
    test_cases = [
        {
            "name": "concat_axis",
            "axes": 0
        },
        {
            "name": "concat_minus_axis",
            "axes": -1
        }
    ]
    for test in test_cases:
        pdpd_concat(test["name"], x, y, test["axes"])
