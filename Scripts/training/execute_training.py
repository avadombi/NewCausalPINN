from Scripts.data_pipeline.data_pipeline import execute_data_pipeline
from Scripts.training.delete_files import delete_files
from Scripts.theory_guided_ml.theory_guided_structure import build_model_structure
from Scripts.training.training_loop import training_loop
from Scripts.dependencies.dependencies import CONSTRAINT_TYPE, NUM_CNN_KERNEL_LAYERS
from Scripts.training.training_utils import display_model_constraints
from Scripts.training.config_params import CONFIG


def execute_training_loop(model_type='Conv', cluster_number=2):
    # data pipeline
    in_outs = execute_data_pipeline(cluster_number=cluster_number)
    id_wells = in_outs['id_wells']

    training_set = in_outs['train']
    test_set = in_outs['test']

    units = test_set['outputs'].shape[-1]

    # machine learning pipeline
    train_params = {'num_folds': 10, 'lr': 5e-3, 'epochs': 80, 'batch_size': 64}  # 64, 128

    # 1. hyper params
    config = CONFIG[model_type]
    hyper_params = config['hyper_params']
    hyper_params['tgl_params']['units'] = units
    unit_dense = hyper_params['unit_dense']
    const_type = hyper_params['const_type']
    path_save = config['path_save']

    # 1.1. check that the total number of kernel layers in CNN satisfy conditions of SCRC 2/3 for SCRC 2/3
    if const_type in ['SCRC2', 'SCRC3']:
        num_cnn_kernel_layers = 1 + len(unit_dense) + 1  # conv + hidden dense + linear dense
        if num_cnn_kernel_layers not in NUM_CNN_KERNEL_LAYERS[const_type]:
            raise ValueError(f'{num_cnn_kernel_layers} kernel layers is not admitted for {const_type}')

    # delete existing files
    delete_files(path_save)

    # train model
    model = build_model_structure(hyper_params, path_save)
    print(model.summary())
    # display_model_constraints(model)
    # training_loop(model, training_set, test_set, train_params, path_save, id_wells, num_points=100)


to_exec = True
if to_exec:
    execute_training_loop(model_type='Global', cluster_number=2)
else:
    const_type = CONSTRAINT_TYPE[0]
    path_save = '../../Results/Conv' if const_type is None else f'../../Results/{const_type}'
    delete_files(path_save)
