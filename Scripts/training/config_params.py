
CONV = {
    'hyper_params': {
        'is_conventional': True,
        'lin_hydro_model': True,
        'seq': 80,
        'tgl_params': {'units': None, 'mode': 'normal', 'inter_vars': 'qg'},
        'cnn_params': {'ft': 5, 'ks': 11, 'act': 'tanh'},
        'unit_dense': (10,) * 3,
        'const_type': None
    },
    'path_save': '../../Results/Conv'
}

CRC = {
    'hyper_params': {
        'is_conventional': False,
        'lin_hydro_model': True,
        'seq': 80,
        'tgl_params': {'units': None, 'mode': 'normal', 'inter_vars': 'qg'},
        'cnn_params': {'ft': 5, 'ks': 11, 'act': 'tanh'},
        'unit_dense': (10,) * 1,
        'const_type': 'CRC'
    },
    'path_save': '../../Results/CRC'
}

GLOBAL = {
    'hyper_params': {
        'is_conventional': False,
        'lin_hydro_model': True,
        'seq': 80,
        'tgl_params': {'units': None, 'mode': 'normal', 'inter_vars': 'qg'},
        'cnn_params': {'ft': 5, 'ks': 11, 'act': 'tanh'},
        'unit_dense': {'CRC': (3,), 'SCRC1': (3,) * 2, 'SCRC2': (3,) * 3, 'SCRC3': (3,) * 5},
        'const_type': ['CRC', 'SCRC1', 'SCRC2', 'SCRC3']
    },
    'path_save': '../../Results/Global'
}

CONFIG = {'Conv': CONV, 'CRC': CRC, 'Global': GLOBAL}
