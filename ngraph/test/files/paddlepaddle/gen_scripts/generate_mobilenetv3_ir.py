from generate_ir import ov_frontend_run
import sys

if __name__ == "__main__":
    user_shapes = {
        "input_layer": "inputs",
        "shapes": {"inputs": [1, 3, 224, 224]}
    }
    ov_frontend_run(sys.argv[1], user_shapes)