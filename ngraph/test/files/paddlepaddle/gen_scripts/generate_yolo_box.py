#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def yolo_box(name : str, x, img_size, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_img_size = pdpd.static.data(name='img_size', shape=img_size.shape, dtype=img_size.dtype)
        boxes, scores = pdpd.vision.ops.yolo_box(node_x,
                                                node_img_size,
                                                anchors=attrs['anchors'],
                                                class_num=attrs['class_num'],
                                                conf_thresh=attrs['conf_thresh'],
                                                downsample_ratio=attrs['downsample_ratio'],
                                                clip_bbox=attrs['clip_bbox'],
                                                name=None, 
                                                scale_x_y=attrs['scale_x_y'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'img_size': img_size},
            fetch_list=[boxes, scores])
        
        # Save inputs in order of ngraph function, to facilite Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        saveModel(name, exe, feedkeys=['x', 'img_size'], fetchlist=[boxes, scores], inputs=[x, img_size], outputs=outs)

    return outs

def onnx_run(x, img_size, onnx_model="../models/yolo_box_test1/yolo_box_test1.onnx"):
    import onnxruntime as rt

    sess = rt.InferenceSession(onnx_model)
    for input in sess.get_inputs():
        print(input.name)
    for output in sess.get_outputs():
        print(output.name)        
    pred_onx = sess.run(None, {"img_size": img_size, "x": x})
    print(pred_onx)
    return pred_onx

def OV_IR_run(x, img_size, path_to_ie_model="../models/yolo_box_test1/fuzzy"):
    from openvino.inference_engine import IECore

    #IE inference
    ie = IECore()
    net = ie.read_network(model=path_to_ie_model + ".xml", weights=path_to_ie_model + ".bin")
    exec_net = ie.load_network(net, "CPU")
    pred_ie = exec_net.infer({"img_size": img_size, "x": x})
    print(type(pred_ie))
    return pred_ie

def OV_frontend_run(x, img_size, path_to_pdpd_model="../models/yolo_box_test1/", user_shapes=[]):
    from ngraph import FrontEndManager # pylint: disable=import-error
    from ngraph import function_to_cnn # pylint: disable=import-error
    from ngraph import PartialShape    # pylint: disable=import-error

    fem = FrontEndManager()
    print('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    print('Initializing new FE for framework {}'.format("pdpd"))
    fe = fem.loadByFramework("pdpd")
    print(fe)
    inputModel = fe.loadFromFile(path_to_pdpd_model)
    try:
        place = inputModel.getPlaceByTensorName('x')
        print(place)
        print(place.isEqual(None))
    except Exception:
        print('Failed to call model API with hardcoded input name "x"')

    if len(user_shapes) > 0:
        for user_shape in user_shapes:
            inputModel.setPartialShape(user_shape['node'], PartialShape(user_shape['shape']))
    nGraphModel = fe.convert(inputModel)

    net = function_to_cnn(nGraphModel)

    from openvino.inference_engine import IECore

    #IE inference
    ie = IECore()
    exec_net = ie.load_network(net, "CPU")
    pred_ie = exec_net.infer({"img_size": img_size, "x": x})
    print(type(pred_ie))
    return pred_ie    

def checker(outs, dump=False):
    if type(outs) is list:
        for i in range(len(outs)):      
            print("output{} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i])
    if type(outs) is dict:
        for i in outs:
            print("output{} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i])            

def print_2Dlike(data, filename):
    with open(filename+".txt", 'w') as f:
        print(data.shape, file=f)
        _data = data.copy()
        _data = np.squeeze(_data)
        _data.reshape(-1, data.shape[len(data.shape)-1])
        print(_data.shape, file=f)
        print("\n[", file=f)
        for j in range(_data.shape[0]):
            line=""
            for i in range(_data.shape[1]):
                line+="{:.2f}".format(_data[j, i]) + "  "
            print(line, file=f)
        print("]\n", file=f)
    f.close()            

def validate(pred_pdpd: list, pred_onx: list, pred_ie: dict, rtol=1e-05, atol=1e-08):
    checker(pred_pdpd)
    checker(pred_onx)
    checker(pred_ie)

    # compare results: IE vs PDPD vs ONNX
    idx = 0
    for key in pred_ie:
        comp1 = np.all(np.isclose(pred_pdpd[idx], pred_onx[idx], rtol=rtol, atol=atol, equal_nan=True))
        #comp1 = np.all(np.isclose([1,2.1], [1,2], rtol=1e-05, atol=1e-08, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp1: 
            print('\033[91m' + "PDPD and ONNX results are different at {} ".format(idx) + '\033[0m')

        comp2 = np.all(np.isclose(pred_pdpd[idx], pred_ie[key], rtol=rtol, atol=atol, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp2:
            print('\033[91m' + "PDPD and IE results are different at {} ".format(idx) + '\033[0m')

        comp3 = np.all(np.isclose(pred_ie[key], pred_onx[idx], rtol=rtol, atol=atol, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp3:
            print('\033[91m' + "ONNX and IE results are different at {} ".format(idx) + '\033[0m')            

        print_2Dlike(pred_pdpd[idx], "pdpd{}".format(idx))
        print_2Dlike(pred_onx[idx], "onnx{}".format(idx))            
        print_2Dlike(pred_ie[key], "ie{}".format(idx))

        if comp1 and comp2 and comp3:
            print('\033[92m' + "PDPD, ONNX and IE results are identical at {} ".format(idx) + '\033[0m')

        idx += 1            


def main():
    # yolo_box
    pdpd_attrs = {
            'anchors': [10, 13, 16, 30, 33, 23],
            'class_num': 2,
            'conf_thresh': 0.5,
            'downsample_ratio': 32,
            'clip_bbox': False, #There is bug in Paddle2ONN where clip_bbox is always ignored.
            'scale_x_y': 1.0
    }

    N = 1
    num_anchors = int(len(pdpd_attrs['anchors'])//2)
    x_shape = (N, num_anchors * (5 + pdpd_attrs['class_num']), 13, 13)
    imgsize_shape = (N, 2)

    data = np.random.random(x_shape).astype('float32')
    data_ImSize = np.random.randint(10, 20, imgsize_shape).astype('int32') 

    # For any change to pdpd_attrs, do -
    # step 1. generate paddle model
    pred_pdpd = yolo_box('yolo_box_test1', data, data_ImSize, pdpd_attrs)

    # step 2. generate onnx model
    # !paddle2onnx --model_dir=../models/yolo_box_test1/ --save_file=../models/yolo_box_test1/yolo_box_test1.onnx --opset_version=10
    import subprocess
    subprocess.run(["paddle2onnx", "--model_dir=../models/yolo_box_test1/", "--save_file=../models/yolo_box_test1/yolo_box_test1.onnx", "--opset_version=12"])
    pred_onx = onnx_run(data, data_ImSize)

    # step 3.a generate OV IR
    # MO or ngraph/frontend/paddlepaddle Fuzzy unit test helper.
    #subprocess.run(["unit-test", "--gtest_filter=*Fuzzy*"])   
    #pred_ie = OV_IR_run(data, data_ImSize)

    # step3.b alternatively, run from frontend API
    pred_ie = OV_frontend_run(data, data_ImSize)

    # step 4. compare 
    # Try different tolerence
    validate(pred_pdpd, pred_onx, pred_ie)
    validate(pred_pdpd, pred_onx, pred_ie, rtol=1e-4, atol=1e-5) 


if __name__ == "__main__":
    main()     