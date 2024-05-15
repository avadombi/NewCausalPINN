from Scripts.dependencies.dependencies import KFold, RANDOM_STATE, pkl
from Scripts.training.training_utils import compute_standardization_params, standardize_inputs, \
    compute_metrics, save_losses, save_shap_values, save_insightfull_data, \
    plot_metrics, plot_observation_simulation, train_model, rnse_loss

from Scripts.dependencies.dependencies import time


def training_loop(model, training_set, test_set, train_params, path_save, id_wells=None, num_points=100):
    """
    params:
        model: machine learning algorithm (or model)
        training_set: training set -> dictionary { inputs, outputs }
        test_set: test set -> dictionary { inputs, outputs }
        train_params: {
            num_folds: number of folder for the k-fold cross-validation method,
            lr: initial learning rate,
            epochs: number of epochs,
            batch_size: mini-batch size
        }
        path_save: path to folder where to save results
    """

    # split data into k cross-validation folds
    k_fold = KFold(
        n_splits=train_params['num_folds'],
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # k-fold training loop
    inputs, outputs = training_set['inputs'], training_set['outputs']
    idx_splits = k_fold.split(X=inputs, y=outputs)

    fold = 1
    for idx_train, idx_valid in idx_splits:
        # fold path
        path_fold = f'{path_save}/{fold}'

        # get training and validation sets
        inp_train, out_train = inputs[idx_train], outputs[idx_train]
        inp_valid, out_valid = inputs[idx_valid], outputs[idx_valid]

        # dataset for metrics computation (before standardization)
        dataset_metrics = {
            'train': (inp_train, out_train),
            'valid': (inp_valid, out_valid),
            'test': (test_set['inputs'], test_set['outputs'])
        }

        # get standardization params of output variables (on training set)
        out_standardization_params = compute_standardization_params(out_train)

        # standardize output variables
        out_train = standardize_inputs(out_train, out_standardization_params)
        out_valid = standardize_inputs(out_valid, out_standardization_params)

        # store training and validation sets in a dictionary
        train_dataset = {
            'train': (inp_train, out_train),
            'valid': (inp_valid, out_valid)
        }

        # start measuring runtime
        start_time = time.time()

        # train the model for the current fold
        model, history = train_model(model, train_dataset, train_params, rnse_loss)

        # compute runtime
        runtime = time.time() - start_time
        print(f"--- runtime: {round(runtime, 2)} seconds ---")

        # compute metrics
        metrics = compute_metrics(model, dataset_metrics, out_standardization_params, id_wells)

        # save insightfull data
        save_insightfull_data(model, history, metrics, path_fold)

        # save shap values
        if fold == 1:
            save_shap_values(
                model,
                dataset_metrics['train'][0],
                dataset_metrics['test'][0],
                id_wells,
                path_fold,
                num_points
            )

        # plot metrics
        plot_metrics(metrics, path_fold, stage='test')

        # plot observations vs simulations
        plot_observation_simulation(
            model, dataset_metrics['test'][0], dataset_metrics['test'][1],
            out_standardization_params, id_wells, path_fold
        )

        # save out_standardization_params
        with open(f'{path_fold}/out_standardization_params.txt', 'wb') as g_file:
            pkl.dump(out_standardization_params, g_file)

        fold += 1
