from Scripts.dependencies.dependencies import np, EPSILON, tf, Callable, Any, Adam, callbacks, pd, he, COLORS
from Scripts.dependencies.dependencies import plt, FONT_SIZE, text, LINE_WIDTH, shap


def display_model_constraints(model):
    for layer in model.layers:
        # check if the layer has weight constraints
        if hasattr(layer, 'kernel_constraint'):
            # get the layer's weight constraints
            kernel_constraints = layer.kernel_constraint

            # display layer weight constraints
            print(f"Layer: {layer.name}")
            print("Weight Constraints:")
            for weight in layer.trainable_weights:
                # check whether the weight has an associated constraint
                if weight.constraint is not None:
                    print(f"- {weight.name}: {weight.constraint}")
            print()


def compute_standardization_params(inputs):
    """
    params:
        inputs: numpy array
    return:
        standardization_params: {
            mean: mean computed on training set,
            stdv: stdv computed on training set
        }
    """

    inputs_mean = np.mean(inputs, axis=0)
    inputs_stdv = np.std(inputs, axis=0)
    return {'mean': inputs_mean, 'stdv': inputs_stdv}


def standardize_inputs(inputs, standardization_params):
    """
    params:
        inputs: numpy array
        standardization_params: {
            mean: mean computed on training set,
            stdv: stdv computed on training set
        }
    return:
        std_inputs: standardized inputs
    """

    return (inputs - standardization_params['mean']) / (standardization_params['stdv'] + EPSILON)


def destandardize_simulation(simulation, standardization_params):
    """
    params:
        simulation: numpy array
        standardization_params: {
            mean: mean computed on training set,
            stdv: stdv computed on training set
        }
    return:
        simulation: destandardized simulation
    """
    return standardization_params['mean'] + (standardization_params['stdv'] + EPSILON) * simulation


def rnse_loss(observation, simulation):
    numerator = tf.reduce_sum(tf.square(observation - simulation), axis=None)
    denominator = tf.reduce_sum(tf.square(observation - tf.reduce_mean(observation, axis=None)))
    return tf.math.divide_no_nan(numerator, denominator)


def train_model(model, train_dataset, train_params, custom_loss: Callable[[Any, Any], Any]):
    """
    params:
        model: machine learning model
        train_dataset: {
            'train': (inp_train, out_train),
            'valid': (inp_valid, out_valid)
        }
        train_params: {
            num_folds: number of folder for the k-fold cross-validation method,
            lr: initial learning rate,
            epochs: number of epochs,
            batch_size: mini-batch size
        }
        custom_loss: custom loss function. Default: rNSE
    """
    # optimizer
    optimizer = Adam(learning_rate=train_params['lr'], epsilon=10E-3)
    model.compile(loss=custom_loss, optimizer=optimizer)  # custom_loss

    # early stopping
    es = callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=0,
        patience=15,
        restore_best_weights=True
    )

    # learning rate reducer
    reduce = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,
        patience=5,
        verbose=1,
        mode='min',
        min_delta=0.005,
        cooldown=0,
        min_lr=train_params['lr'] / 100
    )

    # early stop by nan
    tnan = callbacks.TerminateOnNaN()

    # fit network
    history = model.fit(
        x=train_dataset['train'][0],
        y=train_dataset['train'][1],
        validation_data=train_dataset['valid'],
        epochs=train_params['epochs'],
        verbose=2,
        batch_size=train_params['batch_size'],
        callbacks=[es, reduce, tnan]
    )

    return model, history


def save_losses(history, path_save):
    loss = np.reshape(np.array(history.history['loss']), (-1, 1))
    val_loss = np.reshape(np.array(history.history['val_loss']), (-1, 1))

    try:
        losses = np.concatenate([loss, val_loss], axis=-1)
        losses = pd.DataFrame(loss, columns=['training', 'validation'])
        losses.to_excel(f'{path_save}/losses.xls')
    except Exception as e:
        print("An error occured: ", str(e))


def save_insightfull_data(model, history, metrics, path_save):
    # 1. save the model
    model.save_weights(filepath=f'{path_save}/model')

    # 2. save losses
    save_losses(history, path_save)

    # 3. save metrics
    writer = pd.ExcelWriter(f'{path_save}/metrics.xlsx')

    metrics['train'].to_excel(writer, sheet_name='train', index=False)
    metrics['valid'].to_excel(writer, sheet_name='valid', index=False)
    metrics['test'].to_excel(writer, sheet_name='test', index=False)

    writer.save()
    writer.close()


def predict(model, inputs, standardization_params):
    """
    params:
        model: trained model
        inputs: inputs data
        standardization_params: { mean, std } (outputs data)
    return
        prediction
    """

    return destandardize_simulation(model.predict(inputs), standardization_params)


def get_metric_given_dataset(model, in_outs, standardization_params, id_wells):
    """
    params:
        model: trained model
        in_outs: (inputs, observations)
        standardization_params: { mean, stdv }
        id_wells: id of obs. wells
    return metrics
    """
    # number of wells
    num_wells = in_outs[-1].shape[-1]
    nse, kge, r, rmse, pbias = [], [], [], [], []

    # get inputs and observations
    X_inputs, Y_obs = in_outs[0], in_outs[-1]

    # predict
    Y_sim = predict(model, X_inputs, standardization_params)

    for well in range(num_wells):
        # get predictions and observations for well number `well`
        predictions, observations = Y_sim[:, well], Y_obs[:, well]

        # compute metrics
        _nse = he.evaluator(he.nse, predictions, observations)
        _kge, _r, _, _ = he.evaluator(he.kge, predictions, observations)
        _rmse = he.evaluator(he.rmse, predictions, observations)
        _pbias = he.evaluator(he.pbias, predictions, observations)

        # store
        nse.append(_nse[0])
        kge.append(_kge[0])
        r.append(_r[0])
        rmse.append(_rmse[0])
        pbias.append(_pbias[0])

    # metrics
    metrics = {'id': id_wells, 'nse': nse, 'kge': kge, 'r': r, 'rmse': rmse, 'pbias': pbias}

    # store as a dataframe
    metrics = pd.DataFrame(metrics)
    return metrics


def compute_metrics(model, dataset_metrics, standardization_params, id_wells=None):
    """
    params:
        model: trained model
        dataset_metrics: {
            'train': (x, y),
            'valid': (x, y),
            'test': (x, y)
        }
        standardization_params: { mean, stdv }
        id_wells: id of obs. wells
    return metrics
    """

    # number of wells
    num_wells = dataset_metrics['train'][-1].shape[-1]

    # if id_wells is None, attribute integer id [1, ..., num_wells]
    id_wells = id_wells if not (id_wells is None) else list(range(1, num_wells + 1))

    # metrics for training stage
    metrics_train = get_metric_given_dataset(
        model, dataset_metrics['train'], standardization_params, id_wells
    )

    # metrics for validation stage
    metrics_valid = get_metric_given_dataset(
        model, dataset_metrics['valid'], standardization_params, id_wells
    )

    # metrics for testing stage
    metrics_test = get_metric_given_dataset(
        model, dataset_metrics['test'], standardization_params, id_wells
    )

    # store in a dict
    metrics = {
        'train': metrics_train,
        'valid': metrics_valid,
        'test': metrics_test
    }

    return metrics


def markup_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_edgecolor(COLORS['gray'])
    ax.spines['left'].set_edgecolor(COLORS['gray'])


def plot_metrics(metrics, path_save, stage='test'):
    plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': 'Arial'})
    fig, ax = plt.subplots()

    # get kge and rmse
    kge = np.round(metrics[stage]['kge'].values[0], 2)
    rmse = np.round(metrics[stage]['rmse'].values[0], 2)

    # boxplot
    boxes = ax.boxplot(x=metrics[stage]['nse'].values)

    # mark-up the axis
    markup_axis(ax)

    for line in boxes['medians']:
        # top of median line
        x, y = line.get_xydata()[1]

        # overlay median value
        text(x + 0.02, y, '%.2f' % y, fontsize=FONT_SIZE, verticalalignment='center')

    plt.xticks([1], [stage])
    plt.ylabel('NSE [-]')
    plt.title('KGE: ' + str(kge) + ', RMSE: ' + str(rmse) + '%')
    plt.tight_layout()
    plt.savefig(f'{path_save}/nse.png', dpi=350)
    plt.clf()


def plot_observation_simulation(model, inputs, observations, standardization_params, names, path_save):
    # predictions
    predictions = predict(model, inputs, standardization_params)

    # plot
    plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': 'Arial'})
    plt.figure(figsize=(20, 10))

    rows, cols = 3, 3
    idx_n, k = 1, 1

    out_dim = predictions.shape[-1]
    save_name = 'OS_' + str(k) + '.png'

    for j in range(out_dim):
        name = names[j]

        plt.subplot(rows, cols, idx_n)
        plt.plot(observations[:, j], color=COLORS['black'], linewidth=LINE_WIDTH)
        plt.plot(predictions[:, j], color=COLORS['orange'], linewidth=LINE_WIDTH)
        plt.xlabel('Time [days]')
        plt.ylabel('GWL [m]')
        plt.title(name)

        if (idx_n == rows * cols) or (j == out_dim - 1):
            idx_n = 0
            k += 1

            plt.tight_layout()
            plt.savefig(f'{path_save}/{save_name}', dpi=300)
            save_name = 'OS_' + str(k) + '.png'
            plt.clf()

        idx_n += 1


def save_shap_values(model, inputs_train, inputs_test, id_wells, path_save, num_points=100):
    # background data
    background = inputs_train[:num_points, :, :]

    # foreground data
    foreground = inputs_test[:num_points, :, :]

    x_shp_reshaped = foreground.reshape(-1, foreground.shape[-1])
    n_size = int(foreground.shape[-1] / 2)

    # shape value
    explainer = shap.DeepExplainer(model=model, data=background)
    shap_value = explainer.shap_values(X=foreground, check_additivity=False)
    print('Shap length: ' + str(len(shap_value)))

    labels = []
    # no_feats = int(x_test.shape[-1] / 2)
    for i in range(2 * n_size):
        if i < n_size:
            if i + 1 < 10:
                labels.append('vi 0' + str(i + 1))
            else:
                labels.append('vi ' + str(i + 1))
        else:
            if (i - n_size + 1) < 10:
                labels.append('ep 0' + str(i - n_size + 1))
            else:
                labels.append('ep ' + str(i - n_size + 1))

    for j in range(n_size):
        shap_reshape = shap_value[j].reshape(-1, shap_value[j].shape[-1])

        # plot
        plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': 'Arial'})
        shap.summary_plot(
            shap_reshape,
            x_shp_reshaped,
            feature_names=labels,
            show=False,
            max_display=8
        )
        plt.xlabel("SHAP value (impact on GWL)")
        plt.title(id_wells[j])
        plt.tight_layout()
        plt.savefig(f'{path_save}/shap{j}.png', dpi=300)
        plt.clf()
