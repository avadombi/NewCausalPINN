from Scripts.dependencies.dependencies import np


def fill_na(inputs):
    """
    params:
        inputs: {vi, ep, phi}
            vi: vertical inflow
            ep: potential evapotranspiration
            phi: groundwater level
    return:
        inputs: {vi, ep, phi} with nan filled by 0
    """
    inputs['vi'].fillna(method='ffill', inplace=True)
    inputs['ep'].fillna(method='ffill', inplace=True)

    # check if 'phi' is a key in inputs dict
    if 'phi' in inputs:
        inputs['phi'].fillna(method='ffill', inplace=True)
    return inputs


def to_numpy_and_concat(inputs):
    """
    params:
        inputs: {vi, ep, phi}
            vi: vertical inflow
            ep: potential evapotranspiration
            phi: groundwater level (eventually)
    return:
        outputs: {inputs: (vi, ep), outputs: phi}
    """

    # to numpy
    vi = inputs['vi'].to_numpy()
    ep = inputs['ep'].to_numpy()

    # reshape to 2d (ensure because it already 2d but...)
    vi = vi.reshape((-1, vi.shape[-1]))
    ep = ep.reshape((-1, ep.shape[-1]))

    # concat
    in_vars = np.concatenate([vi, ep], axis=-1)

    # outputs
    outputs = {'id_wells': inputs['id_wells'], 'inputs': in_vars}

    # check if 'phi' is a key in inputs dict
    if 'phi' in inputs:
        phi = inputs['phi'].to_numpy()
        phi = phi.reshape((-1, phi.shape[-1]))

        outputs['outputs'] = phi

    return outputs


def format_outputs(outputs, seq):
    """
    params:
        outputs: GWL, i.e. phi (n, k)
        seq: sequence length
    return
        phi: (p, seq, k)
    """

    size = outputs.shape[0]
    phi = []
    for j in range(size):
        end_idx = j + seq
        if end_idx >= size:
            break

        seq_y = outputs[end_idx, :]
        phi.append(seq_y)

    # to numpy
    phi = np.array(phi)
    return phi


def process_inputs(inputs, seq):
    """
    params:
        inputs: {id_wells, vi, ep, phi}
            vi: vertical inflow
            ep: potential evapotranspiration
            phi: groundwater level
        seq: sequence length
    return:
        in_out: {t, inputs, outputs}
            where
                t: time variable
                inputs: (p, seq, m)
                outputs: (p, k)
                    p: number of time points
                    k: number of wells
    """
    # 1. fill nan
    inputs = fill_na(inputs)

    # 2. remove time variable in vi
    t = inputs['vi'].pop('t')

    # 3. convert to numpy (vi, ep, phi eventually) the concat vi and ep
    outputs = to_numpy_and_concat(inputs)

    # 4. inputs from (n, m) to (p, seq, m) where p = n - seq + 1
    # same for outputs
    # 4.1. inputs
    window_width = outputs['inputs'].shape[1]  # m
    inputs = np.lib.stride_tricks.sliding_window_view(
        outputs['inputs'], window_shape=(seq, window_width)
    ).squeeze()

    if not (t.iloc[seq:].values.shape[0] == inputs.shape[0]):
        inputs = inputs[:-1, :, :]

    # store the input variables in a dictionary
    in_out = {'t': t.iloc[seq:].values, 'id_wells': outputs['id_wells'], 'inputs': inputs}

    # 4.2. outputs
    if 'outputs' in outputs:
        phi = format_outputs(outputs['outputs'], seq)
        in_out['outputs'] = phi

    return in_out
