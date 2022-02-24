import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras import backend as K

from .dataset import get_dataset
from .model import build_model
from .utils import count_data_items, make_results_plot
from .lr_scheduling import get_lr_callback


def train_model_split(data_path, cfg, device="TPU", tpu=None, strategy=None,
                      replicas=8):
    """The training pipeline for the final models. This function trains the
    model by splitting the dataset into a train and test sets.

    Parameters
    ----------
    data_path : str
        The path containing the image records
    cfg : dict
        Parameter dictionary
    device : str
        Device on which the model is trained
    tpu : ??
    strategy : ??
    replicas : int
        Number of replicas used for parallel training
    """
    # split = train_test_split(test_size=cfg["test_size"],
    #                          random_state=cfg["seed"], shuffle=True)
    preds = []

    if device == "TPU":
        if tpu:
            tf.tpu.experimental.initialize_tpu_system(tpu)

    print('#' * 25)
    print('#### TRAINING')

    files_train = tf.io.gfile.glob(
        [data_path + '/train%.2i*.tfrec' % x for x in train_idx])
    np.random.shuffle(files_train)

    files_valid = tf.io.gfile.glob(
        [data_path + '/train%.2i*.tfrec' % x for x in valid_idx])

    train_dataset = get_dataset(files_train, cfg, augment=True, shuffle=True,
                                repeat=True, dim=cfg["input"]["image_size"],
                                batch_size=cfg["modeling"]["batch_size"])
    valid_dataset = get_dataset(files_valid, cfg, augment=False, shuffle=False,
                                repeat=False, dim=cfg["input"]["image_size"])

    # Build model
    K.clear_session()
    with strategy.scope():
        model = build_model(cfg["input"]["image_size"])

    # Model checkpoint callback
    cb1 = tf.keras.callbacks.ModelCheckpoint('final_model.h5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')

    # Learning rate schedule callback
    cb2 = get_lr_callback(cfg["modeling"]["batch_size"])

    # Fitting the model
    steps_per_epoch = count_data_items(files_train) / \
                        cfg["modeling"]["batch_size"] // replicas
    history = model.fit(train_dataset, epochs=cfg["modeling"]["epochs"],
                        callbacks=[cb1, cb2],
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_dataset, verbose=True)

    print('Loading best model...')
    model.load_weights('final_model.h5')

    print('Predicting OOF with TTA...')
    ds_valid = get_dataset(files_valid, cfg, augment=True, repeat=True,
                           shuffle=False, dim=cfg["input"]["image_size"],
                           batch_size=cfg["modeling"]["batch_size"])
    ct_valid = count_data_items(files_valid)
    steps = cfg["post_processing"]["tta_rounds"] * ct_valid / \
            cfg["modeling"]["batch_size"] / 4 / replicas
    pred = model.predict(ds_valid, steps=steps, verbose=True) \
            [:cfg["post_processing"]["tta_rounds"] * ct_valid,]
    preds.append(np.mean(pred.reshape((ct_valid,
                 cfg["post_processing"]["tta_rounds"]), order='F'),
                 axis=1))

    # Plot training
    if cfg["plot_results"]:
        make_results_plot(history, cfg["modeling"]["epochs"], 0,
                            cfg["input"]["image_size"], "Model name")




def train_model_kfold(data_path, cfg, device="TPU", tpu=None, strategy=None,
                      replicas=8):
    """The training pipeline. Note that the models are progressively saved
    after each fold fitting.

    Parameters
    ----------
    data_path : str
        The path containing the image records
    cfg : dict
        Parameter dictionary
    device : str
        Device on which the model is trained
    tpu : ??
    strategy : ???
    replicas : int
        Number of replicas used for parallel training
    """
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []

    skf = KFold(n_splits=cfg["validation"]["n_folds"], shuffle=True,
                random_state=cfg["seed"])
    preds = np.zeros((count_data_items(files_test), 1))

    for fold, (train_idx , valid_idx) in enumerate(skf.split(np.arange(15))):

        if device == 'TPU':
            if tpu:
                tf.tpu.experimental.initialize_tpu_system(tpu)

        print('#' * 25); print('#### FOLD', fold + 1)

        # Generate train and validation sets
        files_train = tf.io.gfile.glob(
            [data_path[fold] + '/train%.2i*.tfrec' % x for x in train_idx])
        np.random.shuffle(files_train)

        files_valid = tf.io.gfile.glob(
            [data_path[fold] + '/train%.2i*.tfrec' % x for x in valid_idx])
        files_test = np.sort(
            np.array(tf.io.gfile.glob(data_path[fold] + '/test*.tfrec')))

        train_dataset = get_dataset(files_train, cfg, augment=True, shuffle=True,
                                    repeat=True, dim=cfg["input"]["image_size"],
                                    batch_size=cfg["modeling"]["batch_size"])
        valid_dataset = get_dataset(files_valid, cfg, augment=False, shuffle=False,
                                    repeat=False, dim=cfg["input"]["image_size"])

        # Build model
        K.clear_session()
        with strategy.scope():
            model = build_model(cfg["input"]["image_size"])

        # Model checkpoint callback
        cb1 = tf.keras.callbacks.ModelCheckpoint(
            'fold-%i.h5' % fold, monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=True, mode='min',
            save_freq='epoch')

        # Learning rate schedule callback
        cb2 = get_lr_callback(cfg["modeling"]["batch_size"])

        # Fitting the model
        print('Training...')
        steps_per_epoch = count_data_items(files_train) / \
                          cfg["modeling"]["batch_size"] // replicas
        history = model.fit(train_dataset, epochs=cfg["modeling"]["epochs"],
                            callbacks=[cb1, cb2],
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_dataset, verbose=True)

        print('Loading best model...')
        model.load_weights('fold-%i.h5' % fold)

        # Predicting out-of-fold using test time augmentation
        print('Predicting OOF with TTA...')
        ds_valid = get_dataset(files_valid, cfg, augment=True, repeat=True,
                               shuffle=False, dim=cfg["input"]["image_size"],
                               batch_size=cfg["modeling"]["batch_size"])
        ct_valid = count_data_items(files_valid)
        steps = cfg["post_processing"]["tta_rounds"] * ct_valid / \
                cfg["modeling"]["batch_size"] / 4 / replicas
        pred = model.predict(ds_valid, steps=steps,
                             verbose=True)\
              [:cfg["post_processing"]["tta_rounds"] * ct_valid,]
        oof_pred.append(np.mean(pred.reshape((ct_valid,
                        cfg["post_processing"]["tta_rounds"]), order='F'),
                        axis=1))

        # Get out-of-fold targets and names
        ds_valid = get_dataset(files_valid, cfg, augment=False, repeat=False,
                               dim=cfg["input"]["image_size"])
        oof_tar.append(np.array([target.numpy() for _, target in iter(
                       ds_valid.unbatch())]))
        oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
        ds = get_dataset(files_valid, cfg, augment=False, repeat=False,
                         dim=cfg["input"]["image_size"])
        oof_names.append(np.array([img_name.numpy().decode("utf-8")
                                   for _, img_name in iter(ds.unbatch())]))

        # Predict test using test time augmentation
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, cfg, augment=True, repeat=True,
                              shuffle=False, dim=cfg["input"]["image_size"],
                              batch_size=cfg["modeling"]["batch_size"] * 4)
        ct_test = count_data_items(files_test)
        steps = cfg["post_processing"]["tta_rounds"] * ct_test / \
                cfg["modeling"]["batch_size"] / 4 / replicas
        pred = model.predict(ds_test, steps=steps,
                             verbose=True) \
               [:cfg["post_processing"]["tta_rounds"] * ct_test,]
        preds[:,0] += np.mean(pred.reshape((ct_test,
                      cfg["post_processing"]["tta_rounds"]), order='F'), axis=1)

        # Report results
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history['val_auc']))
        print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'
              % (fold + 1, oof_val[-1], auc))

        # Plot training
        if cfg["plot_results"]:
           make_results_plot(history, cfg["modeling"]["epochs"], fold,
                             cfg["input"]["image_size"], "Model name")
