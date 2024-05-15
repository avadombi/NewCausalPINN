from Scripts.dependencies.dependencies import K, NonNeg, Conv1D, MaxPool1D, CONSTRAINT_TYPE, tf
from Scripts.dependencies.dependencies import Dropout, Flatten, Dense, Input, Model, keras, pkl

from Scripts.theory_guided_ml.linear_model import LinearModel
from Scripts.theory_guided_ml.hbv_model import HBVModel
from Scripts.theory_guided_ml.linear_global import LinearWeights


class Neg(keras.constraints.Constraint):
    """Constrains weight tensors to be negative"""

    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.), K.floatx())


def cnn_block(s, params, n_outs, const_type=None, unit_dense=()):
    """
    params:
        s: output variables of the theory-guided layer
        params: {
            ft: number of filters,
            ks: kernel size,
            act: activation function
        }
        n_outs: number of wells
        const_type: constraints type. Default: None
        unit_dense: (u1, ..., uN) number of dense layers and the associated number of nodes ui.
                    uN = number of outputs nodes
    """
    assert const_type in CONSTRAINT_TYPE

    # Conv1D
    k_conv = None
    if const_type in ['CRC', 'SCRC2']:
        k_conv = NonNeg()
    elif const_type in ['SCRC1', 'SCRC3']:
        k_conv = Neg()

    x = Conv1D(
        filters=params['ft'],
        kernel_size=params['ks'],
        padding='causal',
        activation=params['act'],
        kernel_constraint=k_conv
    )(s)

    # MaxPool1D, Dropout, and Flatten
    x = MaxPool1D(padding='same')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)

    # Hidden Dense
    num_dense = len(unit_dense)
    k_dense = [NonNeg(), Neg()]
    for j in range(num_dense + 1):  # +1 to include linear dense layer in the loop (but use n_outs)
        # 1. No constraint
        if const_type is None:
            if j == num_dense:  # linear dense
                x = Dense(units=n_outs, activation='linear', kernel_constraint=None)(x)
            else:
                x = Dense(units=unit_dense[j], activation=params['act'], kernel_constraint=None)(x)

        # 2. CRC constraints
        elif const_type == 'CRC':
            if j == num_dense:  # linear dense
                x = Dense(units=n_outs, activation='linear', kernel_constraint=NonNeg())(x)
            else:
                x = Dense(units=unit_dense[j], activation=params['act'], kernel_constraint=NonNeg())(x)

        # 3. SCRC1
        elif const_type == 'SCRC1':
            if j == num_dense:  # linear dense
                x = Dense(units=n_outs, activation='linear', kernel_constraint=Neg())(x)
            else:
                x = Dense(units=unit_dense[j], activation=params['act'], kernel_constraint=Neg())(x)

        # 4. SCR2 and SCR3
        else:
            i = j + 1
            if const_type == 'SCRC2':
                # if i is odd (2n + 1), then negative, else positive
                if j == num_dense:  # linear dense
                    x = Dense(units=n_outs, activation='linear', kernel_constraint=k_dense[i % 2])(x)
                else:
                    x = Dense(units=unit_dense[j], activation=params['act'],
                              kernel_constraint=k_dense[i % 2])(x)
            elif const_type == 'SCRC3':
                # if i is odd (2n + 1), then positive, else negative
                if j == num_dense:  # linear dense
                    x = Dense(units=n_outs, activation='linear', kernel_constraint=k_dense[(i + 1) % 2])(x)
                else:
                    x = Dense(units=unit_dense[j], activation=params['act'],
                              kernel_constraint=k_dense[(i + 1) % 2])(x)

    return x


def model_structure_complete(is_conventional,
                             lin_hydro_model,
                             seq,
                             tgl_params,
                             cnn_params,
                             unit_dense,
                             const_type):
    """
    params:
        is_conventional: True (1D-CNN) or False (Physics-informed model)
        lin_hydro_model: True (LINEAR MODEL) or False (HBV MODEL)
        tgl_params: {
            units: # of wells,
            mode: 'normal' or '...',
            inter_vars: 'qg'
        }
        cnn_params: {
            ft: number of filters,
            ks: kernel size,
            act: activation function
        }
        unit_dense: tuple of units for each dense layer
    """
    # number of inputs and outputs vars
    n_outs = tgl_params['units']
    n_inp = n_outs * 2

    # input layer
    inp = Input(shape=(seq, n_inp))

    if lin_hydro_model:
        x = LinearModel(units=n_outs)(inp)
    else:
        x = HBVModel(
            units=n_outs, mode=tgl_params['mode'], inter_vars=tgl_params['inter_vars']
        )(inp)

    # cnn block
    crc = cnn_block(x, cnn_params, n_outs, const_type[0], unit_dense[const_type[0]])
    sr1 = cnn_block(x, cnn_params, n_outs, const_type[1], unit_dense[const_type[1]])
    sr2 = cnn_block(x, cnn_params, n_outs, const_type[2], unit_dense[const_type[2]])
    sr3 = cnn_block(x, cnn_params, n_outs, const_type[3], unit_dense[const_type[3]])

    # linear dense
    outs = LinearWeights()([crc, sr1, sr2, sr3])

    model = Model(inputs=inp, outputs=outs)
    return model


def model_structure_crc(is_conventional,
                        lin_hydro_model,
                        seq,
                        tgl_params,
                        cnn_params,
                        unit_dense,
                        const_type):
    """
    params:
        is_conventional: True (1D-CNN) or False (Physics-informed model)
        lin_hydro_model: True (LINEAR MODEL) or False (HBV MODEL)
        tgl_params: {
            units: # of wells,
            mode: 'normal' or '...',
            inter_vars: 'qg'
        }
        cnn_params: {
            ft: number of filters,
            ks: kernel size,
            act: activation function
        }
        unit_dense: tuple of units for each dense layer
    """
    # number of inputs and outputs vars
    n_outs = tgl_params['units']
    n_inp = n_outs * 2

    # input layer
    inp = Input(shape=(seq, n_inp))

    if lin_hydro_model:
        x = LinearModel(units=n_outs)(inp)
    else:
        x = HBVModel(
            units=n_outs, mode=tgl_params['mode'], inter_vars=tgl_params['inter_vars']
        )(inp)

    # cnn block
    outs = cnn_block(x, cnn_params, n_outs, const_type, unit_dense)

    model = Model(inputs=inp, outputs=outs)
    return model


def model_structure_cnn(is_conventional,
                        lin_hydro_model,
                        seq,
                        tgl_params,
                        cnn_params,
                        unit_dense,
                        const_type):
    """
    params:
        is_conventional: True (1D-CNN) or False (Physics-informed model)
        lin_hydro_model: True (LINEAR MODEL) or False (HBV MODEL)
        tgl_params: {
            units: # of wells,
            mode: 'normal' or '...',
            inter_vars: 'qg'
        }
        cnn_params: {
            ft: number of filters,
            ks: kernel size,
            act: activation function
        }
        unit_dense: tuple of units for each dense layer
    """
    # number of inputs and outputs vars
    n_outs = tgl_params['units']
    n_inp = n_outs * 2

    # input layer
    inp = Input(shape=(seq, n_inp))

    # cnn block
    outs = cnn_block(inp, cnn_params, n_outs, const_type, unit_dense)
    model = Model(inputs=inp, outputs=outs)
    return model


def build_model_structure(hyper_params, path_save):
    """
    params:
        hyper_params: hyperparameters of the model
        path_save: 1
    """
    # build model structure
    const_type = hyper_params['const_type']

    if const_type is None:
        model = model_structure_cnn(**hyper_params)
    elif isinstance(const_type, str):
        assert const_type == 'CRC'
        model = model_structure_crc(**hyper_params)
    elif isinstance(const_type, list):
        model = model_structure_complete(**hyper_params)
    else:
        raise ValueError('Error on const_type')

    # save hyperparameters
    with open(f'{path_save}/hyper_params.txt', 'wb') as g_file:
        pkl.dump(hyper_params, g_file)

    return model
