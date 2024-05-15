def split_train_test(in_out, perc=0.8):
    """
    params:
        in_out: {t, id_wells, inputs, outputs}
            where
                t: time variable
                id_wells: well's ids
                inputs: (p, seq, m)
                outputs: (p, seq, k)
    return
        in_out: {
            't': t,
            'train': {'inputs': inputs, 'outputs': outputs},
            'test': {'inputs': inputs, 'outputs': outputs}
        }
    """

    end_train = int(in_out['outputs'].shape[0] * perc)

    # train
    in_train = in_out['inputs'][:end_train, :, :]
    out_train = in_out['outputs'][:end_train, :]

    # test
    in_test = in_out['inputs'][end_train:, :, :]
    out_test = in_out['outputs'][end_train:, :]

    # outputs
    in_out = {
        't': in_out['t'],
        'id_wells': in_out['id_wells'],
        'train': {'inputs': in_train, 'outputs': out_train},
        'test': {'inputs': in_test, 'outputs': out_test}
    }

    return in_out
