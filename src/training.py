import json
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras
from custom_callbacks import ReturnBestEarlyStopping
from dataset_tools import preprocess_raw_eeg, ACTIONS, train_generator_with_aug, emd_static_augmentation, \
    load_all_raw_data
from neural_nets import EEGNet
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # shuts down GPU
from src.GAN import generate_synthetic_data

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Comment the following lines in order to occupy all GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def fit_model(train_X: np.ndarray, train_y: np.ndarray, validation_X: np.ndarray, validation_y: np.ndarray,
              network_hyperparameters_dict: dict, aug_hyperparameters_dict: dict, training_name=""):
    F1 = network_hyperparameters_dict['F1']
    D = network_hyperparameters_dict['D']
    F2 = network_hyperparameters_dict['F2']
    learning_rate = network_hyperparameters_dict['learning_rate']
    batch_size = network_hyperparameters_dict['batch_size']
    model_function = network_hyperparameters_dict['network_to_train']
    epochs = network_hyperparameters_dict['epochs']
    metric_to_monitor = network_hyperparameters_dict['metric_to_monitor']
    gan_path = aug_hyperparameters_dict['gan_path']
    gan_attempt_per_sample = aug_hyperparameters_dict['attempt_per_sample']

    if aug_hyperparameters_dict['emd_static_augmentation']:
        train_X, train_y = emd_static_augmentation(train_X, train_y, aug_hyperparameters_dict['emd_augment_mutliplier'],
                                                   aug_hyperparameters_dict['max_imft'])

    if aug_hyperparameters_dict['gan_static_augmentation']:
        num_classes = np.max(train_y) + 1
        num_samples_per_label = int(len(train_X) / num_classes * aug_hyperparameters_dict['gan_augment_multiplier'])

        for label in range(int(np.max(train_y) + 1)):
            generated_data = generate_synthetic_data(gan_path,
                                                     samples_to_generate=num_samples_per_label,
                                                     attempts=gan_attempt_per_sample,
                                                     label=label, latent_dim=50)
            generated_data = generated_data.squeeze(-1)
            train_X = np.vstack([train_X, generated_data])
            train_y = np.hstack([train_y, np.ones(num_samples_per_label) * label])

    train_generator = train_generator_with_aug(train_X, train_y, batch_size=batch_size, **aug_hyperparameters_dict)

    training_name = f"F1:{F1}_D:{D}_F2:{F2}_lr:{learning_rate}{training_name}"
    model = model_function(nb_classes=len(ACTIONS), F1=F1, D=D, F2=F2)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    model_name = model.name
    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_path: Path = Path(f"../{model_name}_{training_start}_{training_name}")
    Path.mkdir(model_path, exist_ok=True)

    callback_lists = get_callback_lists(model_path, metric_to_monitor)

    history = model.fit(
        x=train_generator,
        steps_per_epoch=ceil(train_X.shape[0] / batch_size),
        validation_data=(validation_X, validation_y),
        epochs=epochs,
        callbacks=callback_lists,
    )

    plot_model_accuracy_and_loss(history, model_path)

    save_hyperparameters_dicts(aug_hyperparameters_dict, network_hyperparameters_dict, model_path)


def save_hyperparameters_dicts(aug_hyperparameters_dict, network_hyperparameters_dict, model_path):
    with open(model_path / "augment_parameters.json", 'w+') as file:
        json.dump(aug_hyperparameters_dict, file, sort_keys=True, indent=4)

    model_function = network_hyperparameters_dict['network_to_train']
    network_hyperparameters_dict['network_to_train'] = network_hyperparameters_dict['network_to_train'].__name__
    with open(model_path / "network_parameters.json", 'w+') as file:
        json.dump(network_hyperparameters_dict, file, sort_keys=True, indent=4)
    network_hyperparameters_dict['network_to_train'] = model_function


def plot_model_accuracy_and_loss(history, model_path):
    plt.clf()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_path}/training-validation-loss')

    plt.clf()
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(f'{model_path}/training-validation-accuracy')


def get_callback_lists(model_path, metric_to_monitor):
    callbacks_list = [
        ReturnBestEarlyStopping(
            monitor=f'{metric_to_monitor}',
            patience=4000,
            mode='auto',
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_path}' + '/saved-models-{epoch:06d}-{val_accuracy:.5f}.h5',
            monitor=f'{metric_to_monitor}',
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(
            filename=f'{model_path}/my_model.csv',
            separator=',',
            append=True
        ),
    ]
    return callbacks_list


def kfold_cross_val(data_X: np.ndarray, data_y: np.ndarray, network_hyperparameters_dict: dict,
                    aug_hyperparameters_dict: dict):
    F1 = network_hyperparameters_dict['F1']
    D = network_hyperparameters_dict['D']
    F2 = network_hyperparameters_dict['F2']
    learning_rate = network_hyperparameters_dict['learning_rate']
    batch_size = network_hyperparameters_dict['batch_size']
    model_function = network_hyperparameters_dict['network_to_train']
    epochs = network_hyperparameters_dict['epochs']
    training_name = f"F1:{F1}_D:{D}_F2:{F2}_lr:{learning_rate}"
    random_state = network_hyperparameters_dict['RANDOM_STATE']
    metric_to_monitor = network_hyperparameters_dict['metric_to_monitor']
    gan_path = aug_hyperparameters_dict['gan_path']
    gan_attempt_per_sample = aug_hyperparameters_dict['attempt_per_sample']
    model_name = model_function.__name__
    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_folds = network_hyperparameters_dict['num_folds'] = 10

    models_path: Path = Path(
        f"../Aug_Test_{random_state}") / Path(f"{model_name}_{training_start}_{num_folds}folded_{training_name}")
    Path.mkdir(models_path, exist_ok=True, parents=True)

    save_hyperparameters_dicts(aug_hyperparameters_dict, network_hyperparameters_dict, models_path)

    acc_per_fold = []
    loss_per_fold = []

    strat_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                  random_state=random_state)  # For mantaining class balance
    fold_no = 1

    for train_indexes, test_indexes in strat_kfold.split(data_X, data_y):
        model = model_function(nb_classes=len(ACTIONS), F1=F1, D=D, F2=F2)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        curerent_fold_model_path = Path(models_path / f"fold_n{fold_no}")
        Path.mkdir(curerent_fold_model_path, exist_ok=True)
        callback_lists = get_callback_lists(curerent_fold_model_path, metric_to_monitor)

        train_X, val_X, train_y, val_y = train_test_split(data_X[train_indexes], data_y[train_indexes], test_size=2 / 9,
                                                          random_state=random_state,
                                                          stratify=data_y[train_indexes])
        test_X = data_X[test_indexes]
        test_y = data_y[test_indexes]

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        if aug_hyperparameters_dict['emd_static_augmentation']:
            train_X, train_y = emd_static_augmentation(train_X, train_y,
                                                       aug_hyperparameters_dict['emd_augment_mutliplier'],
                                                       aug_hyperparameters_dict['max_imft'])

        if aug_hyperparameters_dict['gan_static_augmentation']:
            num_classes = np.max(train_y) + 1
            num_samples_per_label = int(len(train_X) / num_classes * aug_hyperparameters_dict['gan_augment_multiplier'])

            for label in range(int(np.max(train_y) + 1)):
                generated_data = generate_synthetic_data(gan_path,
                                                         samples_to_generate=num_samples_per_label,
                                                         attempts=gan_attempt_per_sample,
                                                         label=label, latent_dim=50)
                generated_data = generated_data.squeeze(-1)
                train_X = np.vstack([train_X, generated_data])
                train_y = np.hstack([train_y, np.ones(num_samples_per_label) * label])

        train_generator = train_generator_with_aug(train_X, train_y, batch_size=batch_size, **aug_hyperparameters_dict)
        history = model.fit(train_generator,
                            batch_size=batch_size,
                            steps_per_epoch=ceil(train_X.shape[0] / batch_size),
                            epochs=epochs,
                            verbose=1,
                            validation_data=(val_X, val_y),
                            callbacks=callback_lists)
        scores = model.evaluate(test_X, test_y, verbose=0)

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};'
              f' {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        fold_no += 1

        plot_model_accuracy_and_loss(history, curerent_fold_model_path)

    print_save_kfold_run_results(acc_per_fold, loss_per_fold, models_path)


def print_save_kfold_run_results(acc_per_fold, loss_per_fold, models_path):
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]:.3f} - Accuracy: {acc_per_fold[i]:.3f}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold):.3f} (+- {np.std(acc_per_fold):.3f})')
    print(f'> Loss: {np.mean(loss_per_fold):.3f}')
    print('------------------------------------------------------------------------')
    with open(models_path / f"training_results_acc:{np.mean(acc_per_fold):.3f}", 'w+') as file:
        for i in range(len(acc_per_fold)):
            file.write("----------------------------------------------------\n")
            file.write(f'> Fold {i + 1} - Loss: {loss_per_fold[i]:.3f} - Accuracy: {acc_per_fold[i]:.3f}%\n')
        file.write('------------------------------------------------------------------------\n')
        file.write('Average scores for all folds:\n')
        file.write(f'> Accuracy: {np.mean(acc_per_fold):.3f} (+- {np.std(acc_per_fold):.3f})\n')
        file.write(f'> Loss: {np.mean(loss_per_fold):.3f}\n')
        file.write('------------------------------------------------------------------------')


class Hyperparameters:
    def __init__(self, random_state, label_mapping):
        self.random_state = random_state
        self.label_mapping = label_mapping

    def set_default_hyperparameters(self):
        network_hyperparameters_dict = {}
        aug_hyperparameters_dict = {}

        # NETWORKS PARAMETERS
        network_hyperparameters_dict['network_to_train'] = EEGNet
        network_hyperparameters_dict['epochs'] = 10000
        network_hyperparameters_dict['learning_rate'] = 5e-5
        network_hyperparameters_dict['F1'] = 12
        network_hyperparameters_dict['D'] = 2
        network_hyperparameters_dict['F2'] = 24
        network_hyperparameters_dict['RANDOM_STATE'] = self.random_state
        network_hyperparameters_dict['batch_size'] = 32
        network_hyperparameters_dict['metric_to_monitor'] = 'val_loss'
        network_hyperparameters_dict['label_mapping'] = self.label_mapping
        network_hyperparameters_dict['num_folds'] = 10

        # SCRAMBLING AUGMENTATION PARAMETERS
        aug_hyperparameters_dict['mirror_channel_probability'] = 0
        aug_hyperparameters_dict['shuffle_channel_probability'] = 0
        aug_hyperparameters_dict['shuffle_factor'] = 1

        # EMD AUGMENTATION PARAMETERS
        aug_hyperparameters_dict['emd_sample_probability'] = 0  # For performance issues is recommended maintain
        # such value to zero, use static emd augmentation instead
        aug_hyperparameters_dict['max_imft'] = 6
        aug_hyperparameters_dict['emd_static_augmentation'] = False
        aug_hyperparameters_dict['emd_augment_mutliplier'] = 0

        # GAN AUGMENTATION PARAMETERS
        aug_hyperparameters_dict['gan_static_augmentation'] = False
        aug_hyperparameters_dict['gan_path'] = None
        aug_hyperparameters_dict['gan_augment_multiplier'] = 0
        aug_hyperparameters_dict['attempt_per_sample'] = 500

        # NOISE AUGMENTATION PARAMETER
        aug_hyperparameters_dict['gaussian_noise_std'] = 0

        # STFT NOISE AUGMENTATION PARAMETERS
        aug_hyperparameters_dict['stft_noise_sample_probability'] = 0
        aug_hyperparameters_dict['gaussian_noise_stft_std'] = 1e-2
        aug_hyperparameters_dict['stft_window_size'] = 20

        return network_hyperparameters_dict, aug_hyperparameters_dict


def test_routine(GAN_PATH, data_X, data_y, hyperparameters, fixed_random_state: int = None):
    if fixed_random_state:
        hyperparameters.random_state = fixed_random_state

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['mirror_channel_probability'] = 0.01
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['mirror_channel_probability'] = 0.02
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['mirror_channel_probability'] = 0.05
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['mirror_channel_probability'] = 0.10
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['mirror_channel_probability'] = 0.15
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['shuffle_channel_probability'] = 0.01
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['shuffle_channel_probability'] = 0.02
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['shuffle_channel_probability'] = 0.05
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['shuffle_channel_probability'] = 0.10
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['shuffle_channel_probability'] = 0.15
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['emd_static_augmentation'] = True
    aug_hyperparameters_dict['emd_augment_mutliplier'] = 0.25
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['emd_static_augmentation'] = True
    aug_hyperparameters_dict['emd_augment_mutliplier'] = 0.5
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['emd_static_augmentation'] = True
    aug_hyperparameters_dict['emd_augment_mutliplier'] = 1
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['emd_static_augmentation'] = True
    aug_hyperparameters_dict['emd_augment_mutliplier'] = 2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['emd_static_augmentation'] = True
    aug_hyperparameters_dict['emd_augment_mutliplier'] = 3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gan_static_augmentation'] = True
    aug_hyperparameters_dict['gan_path'] = GAN_PATH
    aug_hyperparameters_dict['gan_augment_multiplier'] = 0.25
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gan_static_augmentation'] = True
    aug_hyperparameters_dict['gan_path'] = GAN_PATH
    aug_hyperparameters_dict['gan_augment_multiplier'] = 0.5
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gan_static_augmentation'] = True
    aug_hyperparameters_dict['gan_path'] = GAN_PATH
    aug_hyperparameters_dict['gan_augment_multiplier'] = 1
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gan_static_augmentation'] = True
    aug_hyperparameters_dict['gan_path'] = GAN_PATH
    aug_hyperparameters_dict['gan_augment_multiplier'] = 2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gan_static_augmentation'] = True
    aug_hyperparameters_dict['gan_path'] = GAN_PATH
    aug_hyperparameters_dict['gan_augment_multiplier'] = 3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gaussian_noise_std'] = 1e-3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gaussian_noise_std'] = 5e-3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gaussian_noise_std'] = 1e-2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gaussian_noise_std'] = 5e-2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['gaussian_noise_std'] = 1e-1
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['stft_noise_sample_probability'] = 1
    aug_hyperparameters_dict['gaussian_noise_stft_std'] = 1e-3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['stft_noise_sample_probability'] = 1
    aug_hyperparameters_dict['gaussian_noise_stft_std'] = 5e-3
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['stft_noise_sample_probability'] = 1
    aug_hyperparameters_dict['gaussian_noise_stft_std'] = 1e-2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['stft_noise_sample_probability'] = 1
    aug_hyperparameters_dict['gaussian_noise_stft_std'] = 5e-2
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    aug_hyperparameters_dict['stft_noise_sample_probability'] = 1
    aug_hyperparameters_dict['gaussian_noise_stft_std'] = 1e-1
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    kfold_cross_val(data_X, data_y, network_hyperparameters_dict, aug_hyperparameters_dict)


def main():
    parser = ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    RANDOM_STATE = args.random_state

    STARTING_DIR = Path("../chris_personal_dataset")
    GAN_PATH = "../WGAN_2021-04-16 00:58:22/models"  # Insert GAN model path here or comment such line
    SPLITTING_PERCENTAGE = namedtuple('SPLITTING_PERCENTAGE', ['train', 'val', 'test'])
    SPLITTING_PERCENTAGE.train, SPLITTING_PERCENTAGE.val, SPLITTING_PERCENTAGE.test = (70, 20, 10)

    raw_data_X, data_y, label_mapping = load_all_raw_data(starting_dir=STARTING_DIR)

    hyperparameters = Hyperparameters(RANDOM_STATE, label_mapping)

    network_hyperparameters_dict, aug_hyperparameters_dict = hyperparameters.set_default_hyperparameters()

    data_X, fft_data_X = preprocess_raw_eeg(raw_data_X, lowcut=8, highcut=45, coi3order=0)

    tmp_train_X, test_X, tmp_train_y, test_y = train_test_split(data_X, data_y,
                                                                test_size=SPLITTING_PERCENTAGE.test / 100,
                                                                random_state=network_hyperparameters_dict[
                                                                    'RANDOM_STATE'], stratify=data_y)
    actual_valid_split_fraction = SPLITTING_PERCENTAGE.val / (100 - SPLITTING_PERCENTAGE.test)
    train_X, val_X, train_y, val_y = train_test_split(tmp_train_X, tmp_train_y, test_size=actual_valid_split_fraction,
                                                      random_state=network_hyperparameters_dict['RANDOM_STATE'],
                                                      stratify=tmp_train_y)

    test_routine(GAN_PATH, data_X, data_y, hyperparameters)


if __name__ == '__main__':
    main()
