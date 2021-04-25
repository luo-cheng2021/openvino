import sys
import os

def ov_frontend_run(path_to_pdpd_model: str, user_shapes: dict):
    from ngraph import FrontEndManager # pylint: disable=import-error
    from ngraph import function_to_cnn # pylint: disable=import-error
    from ngraph import PartialShape    # pylint: disable=import-error

    fem = FrontEndManager()
    print('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    print('Initializing new FE for framework {}'.format("pdpd"))
    fe = fem.loadByFramework("pdpd")
    print(fe)
    full_model_path = os.path.abspath(path_to_pdpd_model)
    print("Prepare to convert ", full_model_path)

    input_model = fe.loadFromFile(full_model_path)
    input_places = input_model.getInputs()

    for place in input_places:
        place_name = place.getNames()[0]
        input_model.setPartialShape(place, PartialShape(user_shapes["shapes"][place_name]))

    model = fe.convert(input_model)

    ie_network = function_to_cnn(model)
    full_model_path = os.path.abspath(path_to_pdpd_model)
    print(full_model_path)
    model_name = full_model_path.split('.')
    print(model_name)
    ie_network.serialize(model_name[0] + ".xml", model_name[0] + ".bin")


if __name__ == "__main__":
    ov_frontend_run(sys.argv[1])