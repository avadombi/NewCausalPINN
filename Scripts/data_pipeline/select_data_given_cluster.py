from Scripts.dependencies.dependencies import pd


def get_data_cluster(path_data='../Data', aquifer='uc', cluster_number=3):
    """
    params:
        path_data: folder where data are located. Default: '../Data/'
        aquifer: type of aquifer. Values: uc (unconfined), sc (semi-confined), c(confined). Default: 'uc'
        cluster_number: cluster id
    return:
        inputs where
            inputs: dictionary of tensors of vi, ep, and phi (GWL)
    """

    assert aquifer in ['uc', 'sc', 'c']

    # get well's id for the chosen cluster
    # and reformat it by adding a '0' at the begining
    data = pd.read_excel(f'{path_data}/clusters.xlsx')
    id_wells = '0' + (data[data['clusters'] == cluster_number]['id']).astype('str')
    id_wells = id_wells.to_list()

    # columns to return
    columns = ['t'] + id_wells

    # 1. load inputs data (vi and ep)
    inputs = pd.ExcelFile(f'{path_data}/inputs.xlsx')
    vi = pd.read_excel(inputs, sheet_name='vi', index_col=0)[columns]
    ep = pd.read_excel(inputs, sheet_name='ep', index_col=0)[id_wells]

    # 2. load outputs data (GWL)
    phi = pd.read_excel(f'{path_data}/reference_head.xlsx')[id_wells]
    inputs = {'id_wells': id_wells, 'vi': vi, 'ep': ep, 'phi': phi}

    return inputs
