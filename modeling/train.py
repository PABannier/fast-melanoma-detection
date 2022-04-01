import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras import backend as K

from .dataset import get_dataset
from .model import build_model
from .utils import count_data_items, make_results_plot


def train_model(data_path, cfg, auto, replicas, strategy):
    """Train model.

    Parameters
    ----------
    data_path : str
        The path containing the image records.

    cfg : dict
        Dictionary containing the training parameters.
    """
    print("#" * 25)
    print("#### TRAINING")
    print("\n")

    oof_pred = []
    oof_val = []
    oof_tar = []

    for valid_fold in range(cfg["n_folds"]):
        files_train = None
        files_valid = None

        np.random.shuffle(files_train)

        train_dataset = get_dataset(files_train, auto, replicas, augment=True,
                                    shuffle=True, repeat=True,
                                    dim=cfg["input"]["image_size"],
                                    batch_size=cfg["train"]["batch_size"])
        valid_dataset = get_dataset(files_valid, auto, replicas, augment=False,
                                    shuffle=False, repeat=False,
                                    dim=cfg["input"]["image_size"])

        # Build model
        K.clear_session()

        with strategy.scope():
            model = build_model(
                dim=cfg["input"]["image_size"], init_lr=cfg["train"]["init_lr"],
                min_lr=cfg["train"]["min_lr"], num_epochs=cfg["train"]["num_epochs"],
                num_classes=cfg["model"]["num_classes"], dropout=cfg["train"]["dropout"],
                weight_decay=cfg["train"]["weight_decay"],
                num_snapshots=cfg["train"]["num_snapshots"],
                multi_sample_dropout=cfg["train"]["multi_sample_dropout"]
            )

        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/fold-%i.h5' % valid_fold, monitor="val_loss", verbose=0,
            save_best_only=True, save_weights_only=True, mode="min", save_freq="epoch"
        )

        # Fitting the model
        steps_per_epoch = count_data_items(files_train) / cfg["train"]["batch_size"] \
                          // replicas
        history = model.fit(train_dataset, epochs=cfg["train"]["num_epochs"],
                            callbacks=[checkpoint], steps_per_epoch=steps_per_epoch,
                            validation_data=valid_dataset, verbose=True)

        print("\n")
        print("#" * 25)
        print("#### VALIDATION")

        model.load_weights("checkpoints/fold-%i.h5" % valid_fold)

        # Predicting OOF
        ds_valid = get_dataset(files_valid, auto, replicas, augment=True, repeat=True,
                               shuffle=False, dim=cfg["input"]["image_size"],
                               batch_size=cfg["train"]["batch_size"])

        ct_valid = count_data_items(files_valid)
        steps = cfg["post_processing"]["tta_rounds"] * ct_valid / \
                cfg["modeling"]["batch_size"] / 4 / replicas

        pred = model.predict(ds_valid, steps=steps, verbose=True)\
            [:cfg["post_processing"]["tta_rounds"] * ct_valid,]
        oof_pred.append(np.mean(pred.reshape((ct_valid,
                        cfg["post_processing"]["tta_rounds"]), order='F'),
                        axis=1))

        # Get OOF targets
        ds_valid = get_dataset(files_valid, cfg, augment=False, repeat=False,
                               dim=cfg["input"]["image_size"])
        oof_tar.append(np.array([target.numpy() for _, target in iter(
                       ds_valid.unbatch())]))

        # Report results
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1], multi_class="ovo",
                            average="macro")
        oof_val.append(np.max(history.history['val_auc']))
        print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'
              % (valid_fold + 1, oof_val[-1], auc))

        # Plot training
        if cfg["plot_results"]:
           make_results_plot(history, cfg["modeling"]["epochs"], valid_fold,
                             cfg["input"]["image_size"], "Model name")
