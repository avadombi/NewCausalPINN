from Scripts.plots.plot_obs_sim import load_model
from Scripts.plots.confi_plot import MODEL_PARAMS
from Scripts.dependencies.dependencies import np
from Scripts.theory_guided_ml.theory_guided_structure import build_model_structure


def display_last_layer_weight():
    # get params
    params = MODEL_PARAMS[2]

    # get model
    model, out_standardization_params = load_model(**params)

    # get the last three layers of the model
    last_layer = model.layers[-1]

    # print the weight values of the last layer
    weights = last_layer.get_weights()
    print(f"Layer: {last_layer.name}")
    for i, weight in enumerate(weights):
        print(f"Weight {i}: {np.round(weight, 3)}")
    print()


def display_num_params():
    # params
    config = {
        'hyper_params': {
            'is_conventional': True,
            'lin_hydro_model': True,
            'seq': 80,
            'tgl_params': {'units': None, 'mode': 'normal', 'inter_vars': 'qg'},
            'cnn_params': {'ft': 5, 'ks': 11, 'act': 'tanh'},
            'unit_dense': (3,) * 5,
            'const_type': None
        },
        'path_save': '../../Results/Conv'
    }

    hyper_params = config['hyper_params']
    hyper_params['tgl_params']['units'] = 6
    unit_dense = hyper_params['unit_dense']
    const_type = hyper_params['const_type']
    path_save = config['path_save']

    # build a new model with chosen architecture
    model = build_model_structure(hyper_params, path_save)

    # print
    counter = 0
    for layer in model.layers:
        num_params = layer.count_params()
        counter += num_params
        print(f"{layer.name}: {num_params} - Cumuli: {counter}")


display_num_params()
