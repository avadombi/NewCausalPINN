from Scripts.data_pipeline.select_data_given_cluster import get_data_cluster
from Scripts.data_pipeline.process_data import process_inputs
from Scripts.data_pipeline.split_data import split_train_test


def execute_data_pipeline(cluster_number=2):
    in_outs = split_train_test(                # 3. split data into training and test sets
        process_inputs(                        # 2. process data
            get_data_cluster(                  # 1. select data for a given cluster
                path_data='../../Data',
                aquifer='uc',
                cluster_number=cluster_number
            ),
            seq=80
        ),
        perc=0.8
    )
    return in_outs
