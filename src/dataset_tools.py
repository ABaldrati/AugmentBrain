# File heavily based on https://github.com/CrisSherban/BrainPad
import math
import os
import shutil
from pathlib import Path
from statistics import mean

import emd
import numpy as np
from brainflow import DataFilter, FilterTypes
from tensorflow.keras.utils import Sequence
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import stft, istft
from tqdm import tqdm

BOARD_SAMPLING_RATE = 250
ACTIONS = ["feet", "none", "hands"]


def check_std_deviation(sample: np.ndarray, lower_threshold=0.01, upper_threshold=25):
    stds = []
    for i in range(len(sample)):
        std = sample[i].std()
        stds.append(int(std))
        print(f"{i} - {std}")
    for i in range(len(sample)):
        std = sample[i].std()
        if std < lower_threshold:
            print("An electrode may be disconnected")
            return False
        if std > upper_threshold:
            print(f"Noisy_sample, channel{i} - {std}")
    print(f"average std deviation: {mean(stds)}")
    while True:
        input_save = input("Do you want to save this sample? [Y,n]")
        if 'n' in input_save:
            return False
        elif 'y' in input_save.lower() or '' == input_save:
            return True
        else:
            print("Enter 'y' to save the sample or 'n' to discard the sample")


def split_data(starting_dir="../personal_dataset", splitting_percentage=(70, 20, 10), shuffle=True, coupling=False,
               division_factor=0):
    """
        This function splits the dataset in three folders, training, validation, untouched
        Has to be run just everytime the dataset is changed

    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, untouched_percentage)
    :param shuffle: bool, decides if the personal_dataset will be shuffled
    :param coupling: bool, decides if samples are shuffled singularly or by couples
    :param division_factor: int, if the personal_dataset used is made of FFTs which are taken from multiple sittings
                            one sample might be very similar to an adjacent one, so not all the samples
                            should be considered because some very similar samples could fall both in
                            validation and training, thus the division_factor divides the personal_dataset.
                            if division_factor == 0 the function will maintain all the personal_dataset

    """
    training_per, validation_per, untouched_per = splitting_percentage

    # if not os.path.exists("../training_data") and not os.path.exists("../validation_data") \
    #         and not os.path.exists("../untouched_data"):
    if os.path.exists("../training_data"):
        shutil.rmtree("../training_data")
    if os.path.exists("../validation_data"):
        shutil.rmtree("../validation_data")
    if os.path.exists("../untouched_data"):
        shutil.rmtree("../untouched_data")

    os.mkdir("../training_data")
    os.mkdir("../validation_data")
    os.mkdir("../untouched_data")

    for action in ACTIONS:

        action_data = []
        all_action_data = []
        # this will contain all the samples relative to the action

        data_dir = os.path.join(starting_dir, action)
        # sorted will make sure that the personal_dataset is appended in the order of acquisition
        # since each sample file is saved as "timestamp".npy
        for file in sorted(os.listdir(data_dir)):
            # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
            all_action_data.append(np.load(os.path.join(data_dir, file)))

        # TODO: make this coupling part readable
        if coupling:
            # coupling near time acquired samples to reduce the probability of having
            # similar samples in both train and validation sets
            coupled_actions = []
            first = True
            for i in range(len(all_action_data)):
                if division_factor != 0:
                    if i % division_factor == 0:
                        if first:
                            tmp_act = all_action_data[i]
                            first = False
                        else:
                            coupled_actions.append([tmp_act, all_action_data[i]])
                            first = True
                else:
                    if first:
                        tmp_act = all_action_data[i]
                        first = False
                    else:
                        coupled_actions.append([tmp_act, all_action_data[i]])
                        first = True

            if shuffle:
                np.random.shuffle(coupled_actions)

            # reformatting all the samples in a single list
            for i in range(len(coupled_actions)):
                for j in range(len(coupled_actions[i])):
                    action_data.append(coupled_actions[i][j])

        else:
            for i in range(len(all_action_data)):
                if division_factor != 0:
                    if i % division_factor == 0:
                        action_data.append(all_action_data[i])
                else:
                    action_data = all_action_data

            if shuffle:
                np.random.shuffle(action_data)

        num_training_samples = int(len(action_data) * training_per / 100)
        num_validation_samples = int(len(action_data) * validation_per / 100)
        num_untouched_samples = int(len(action_data) * untouched_per / 100)

        # creating subdirectories for each action
        tmp_dir = os.path.join("../training_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(num_training_samples):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

        tmp_dir = os.path.join("../validation_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(num_training_samples, num_training_samples + num_validation_samples):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

        if untouched_per != 0:
            tmp_dir = os.path.join("../untouched_data", action)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for sample in range(num_training_samples + num_validation_samples,
                                num_training_samples + num_validation_samples + num_untouched_samples):
                np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])


def load_all_raw_data(starting_dir: Path, channels=8, NUM_TIMESTAMP_PER_SAMPLE=250):
    data_X = np.empty((0, channels, NUM_TIMESTAMP_PER_SAMPLE))
    data_y = np.empty(0)
    mapping = {}
    filtered_actions = [action_dir for action_dir in starting_dir.iterdir() if action_dir.name in ACTIONS]
    for index, actions_dir in enumerate(filtered_actions):
        if actions_dir.name in ACTIONS:
            for sample_path in actions_dir.iterdir():
                data_X = np.append(data_X, np.expand_dims(np.load(str(sample_path)), axis=0), axis=0)
                data_y = np.append(data_y, index)
                mapping[index] = actions_dir.name
    return data_X, data_y, mapping


def load_data(starting_dir, shuffle=True, balance=False):
    """
        This function loads the personal_dataset from a directory where the classes
        have been split into different folders where each file is a sample

    :param starting_dir: the path of the personal_dataset you want to load
    :param shuffle: bool, decides if the personal_dataset will be shuffled
    :param balance: bool, decides if samples should be equal in cardinality between classes
    :return: X, y: both python lists
    """

    data = [[] for i in range(len(ACTIONS))]
    for i, action in enumerate(ACTIONS):

        data_dir = os.path.join(starting_dir, action)
        for file in sorted(os.listdir(data_dir)):
            data[i].append(np.load(os.path.join(data_dir, file)))

    if balance:
        lengths = [len(data[i]) for i in range(len(ACTIONS))]

        # this is required if one class has more samples than the others
        for i in range(len(ACTIONS)):
            data[i] = data[i][:min(lengths)]

        lengths = [len(data[i]) for i in range(len(ACTIONS))]

    # this is needed to shuffle the personal_dataset between classes, so the model
    # won't train first on one single class and then pass to the next one
    # but it trains on all classes "simultaneously"
    combined_data = []

    # we are using one hot encodings
    for i in range(len(ACTIONS)):
        lbl = np.zeros(len(ACTIONS), dtype=int)
        lbl[i] = 1
        for sample in data[i]:
            combined_data.append([sample, lbl])

    if shuffle:
        np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)


def standardize(data, std_type="channel_wise"):
    if std_type == "feature_wise":
        for j in range(len(data[0, 0, :])):
            mean = data[:, :, j].mean()
            std = data[:, :, j].std()
            data[:, :, j] = (data[:, :, j] - mean) / std

    if std_type == "sample_wise":
        for k in range(len(data)):
            mean = data[k].mean()
            std = data[k].std()
            data[k] -= mean
            data[k] /= std

    if std_type == "channel_wise":
        # this type of standardization prevents some channels to have more importance over others,
        # i.e. back head channels have more uVrms because of muscle tension in the back of the head
        # this way we prevent the network from concentrating too much on those features
        for k in range(len(data)):
            sample = data[k]
            for i in range(len(sample)):
                mean = sample[i].mean()
                std = sample[i].std()
                if std < 0.001:
                    data[k, i, :] = (data[k, i, :] - mean) / (std + 0.1)
                else:
                    data[k, i, :] = (data[k, i, :] - mean) / std

    return data


def visualize_save_data(data, title, visualize_data=True, file_path=None, save_data=False):
    # takes a look at the personal_dataset
    if save_data and file_path is None:
        raise ValueError("When saving data you must define a file path")
    for i in range(len(data)):
        plt.plot(np.arange(len(data[i])), data[i])

    plt.title(title)
    if save_data:
        plt.savefig(file_path)
    if visualize_data:
        plt.show()
    plt.clf()


def visualize_all_data(data):
    # takes a look at the personal_dataset
    for sample in data:
        for i in range(len(sample)):
            plt.plot(np.arange(len(sample[i])), sample[i])
        plt.show()
        plt.clf()


def preprocess_raw_eeg(data, fs=250, lowcut=2.0, highcut=65.0, MAX_FREQ=60, power_hz=50, coi3order=3):
    """
        Processes raw EEG personal_dataset, filters 50Hz noise from electronics in EU, applies bandpass
        and wavelet denoising.
        Change power_hz to 60Hz if you are in the US
        Check local power line frequency otherwise
    :param data: ndarray, input dataset in to filter with shape=(samples, channels, values)
    :param fs: int, sampling rate
    :param lowcut: float, lower extreme for the bandpass filter
    :param highcut: float, higher extreme for the bandpass filter
    :param MAX_FREQ: int, maximum frequency for the FFTs
    :return: tuple, (ndarray, ndarray), process personal_dataset and FFTs respectively
    """
    # print(personal_dataset.shape)
    # visualize_save_data(data[0],
    #                     file_path="../pictures/before",
    #                     title="RAW EEGs",
    #                     save_data=True)

    data = standardize(data)

    # visualize_save_data(data,
    #                     file_path="../pictures/after_std",
    #                     title="After Standardization",
    #                     save_data=True)

    fft_data = np.zeros((len(data), len(data[0]), MAX_FREQ))

    for sample in range(len(data)):
        for channel in range(len(data[0])):
            DataFilter.perform_bandstop(data[sample][channel], sampling_rate=fs, center_freq=power_hz, band_width=2.0,
                                        order=5, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)

            if coi3order != 0:
                DataFilter.perform_wavelet_denoising(data[sample][channel], 'coif3', coi3order)

            DataFilter.perform_bandpass(data[sample][channel], sampling_rate=fs,
                                        center_freq=int((lowcut + highcut) / 2), band_width=highcut - lowcut, order=5,
                                        filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)

            fft_data[sample][channel] = np.abs(fft(data[sample][channel])[:MAX_FREQ])

    fft_data = standardize(fft_data)

    # visualize_save_data(data[0],
    #                     file_path="../pictures/after_bandpass",
    #                     title=f'After bandpass from {lowcut}Hz to {highcut}Hz',
    #                     save_data=True)
    #
    # visualize_save_data(fft_data[0],
    #                     file_path="../pictures/ffts",
    #                     title="FFTs",
    #                     save_data=True)

    return data, fft_data


def emd_static_augmentation(train_X: np.ndarray, train_y: np.ndarray, augment_multiplier=1, max_imft=6,
                            gaussian_noise_std=0):
    print('augmenting dataset using empirical mode decomposition')
    num_classes = int(np.max(train_y) + 1)
    num_channels = train_X.shape[1]
    indices_list = [np.where(train_y == i)[0] for i in range(num_classes)]
    classes_bar = tqdm(range(num_classes), desc="classes")
    for class_index in classes_bar:
        num_class_augmented_sample = int(len(indices_list[class_index]) * augment_multiplier)
        samples_bar = tqdm(range(num_class_augmented_sample), leave=False,
                           desc=f"sample in class:{class_index}/{len(classes_bar) - 1}")
        for _ in samples_bar:
            augmented_sample = np.zeros(train_X[0].shape)
            for imf_index in range(max_imft):
                current_imf_sample_index = np.random.choice(indices_list[class_index])
                current_imf_sample = train_X[current_imf_sample_index]
                for channel in range(num_channels):
                    channel_imf = emd.sift.sift(current_imf_sample[channel])
                    try:
                        augmented_sample[channel] += channel_imf[:, imf_index]
                    except IndexError:
                        pass
            train_X = np.append(train_X, np.expand_dims(augmented_sample, axis=0), axis=0)
            train_y = np.append(train_y, class_index)

    shuffle_indices = np.arange(len(train_X))
    np.random.shuffle(shuffle_indices)
    train_X = train_X[shuffle_indices]
    train_y = train_y[shuffle_indices]

    gaussian_noise_matrix = np.random.normal(loc=0, scale=gaussian_noise_std, size=train_X.shape)
    train_X = train_X + gaussian_noise_matrix

    return train_X, train_y


def add_noise_on_stft_sample_data(sample: np.ndarray, fs: float, window_size: int, gaussian_noise_std: float):
    _, _, Zxx = stft(sample, fs=fs, nperseg=window_size)
    Zxx_magnitude = np.abs(Zxx)
    Zxx_angle = np.angle(Zxx)

    gaussian_noise_matrix = np.random.normal(loc=0, scale=gaussian_noise_std, size=Zxx.shape)
    perturbed_Zxx_magnitude = Zxx_magnitude + gaussian_noise_matrix

    perturbed_Zxx = perturbed_Zxx_magnitude * (np.cos(Zxx_angle) + 1j * np.sin(Zxx_angle))

    _, perturbed_sample = istft(perturbed_Zxx)
    if perturbed_sample.shape != sample.shape:
        raise RuntimeError("Shape mismatch, check windows_size of STFT")
    return perturbed_sample


def check_duplicate(train_X, test_X):
    """
        Checks if there is leaking from the splitting procedure
    :param train_X: ndarray, the training set
    :param test_X:  ndarray, the test set
    :return: bool, True if there is some leaking, False otherwise
    """
    # TODO: find a less naive and faster alternative
    print("Checking duplicated samples split-wise...")

    tmp_train = np.array(train_X)
    tmp_test = np.array(test_X)

    for i in tqdm(range(len(tmp_train))):
        for j in range(len(tmp_test)):
            if np.array_equiv(tmp_train[i], tmp_test[j]):
                print("\n Duplication found! check the splitting procedure")
                return True
    print("You don't have duplicated samples")
    return False


def train_generator_with_aug(train_X: np.ndarray, train_y: np.ndarray, batch_size: int,
                             shuffle_channel_probability=0., mirror_channel_probability=0.,
                             stft_noise_sample_probability=0., shuffle_factor=0, gaussian_noise_std=0.,
                             gaussian_noise_stft_std=0., stft_window_size=20, emd_sample_probability=0., max_imft=6,
                             **kwargs):
    """
    Yield a batch of augmented data, there are few possible data augmentation which are:
    - swap randomly a channel of a sample with another with the same label
    - mirror randomly a channel
    - add gaussian noise on data
    - add gaussian noise on stft transform of original data
    - generate new data with empirical mode decomposition

    :param train_X: the training set
    :param train_y: the training labels
    :param batch_size: the batch size
    :param shuffle_channel_probability: probability of shuffling each channel of a sample, if 1 always shuffle if 0 never
    :param mirror_channel_probability: probability of mirroring each channel of a sample, if 1 always shuffle if 0 never
    :param stft_noise_sample_probability: probability of adding noise on stft for a sample
    :param shuffle_factor: maximum samples shuffled with the principal one
    :param gaussian_noise_std: gaussian noise standard deviation add to data
    :param gaussian_noise_stft_std: gaussian noise standard deviation add stft of data
    :param stft_window_size: short time fourier transform windows size
    :param emd_sample_probability: probability of building a new sample with empirical mode decomposition, if 1 always shuffle if 0 never
    :param max_imft: maximum imft in empirical mode decomposition



    :yield: tuple, (ndarray, ndarray) batch training data and labels
    """
    num_channels = train_X.shape[1]
    num_classes = int(np.max(train_y) + 1)
    indices_lists = [np.where(train_y == i)[0] for i in range(num_classes)]
    index_to_class_dict = {}
    for class_index, class_label_list in enumerate(list(indices_lists)):
        for sample_index in class_label_list:
            index_to_class_dict[sample_index] = class_index

    while True:
        epoch_indexes = np.arange(len(train_X))
        np.random.shuffle(epoch_indexes)
        batch_train_X = np.empty((0, train_X.shape[1], train_X.shape[2]))
        batch_train_y = np.empty(0)
        for principal_sample_index in epoch_indexes:
            if np.random.random() < emd_sample_probability:
                augmented_sample = np.zeros((1, train_X.shape[1], train_X.shape[2]))
                class_index = index_to_class_dict[principal_sample_index]
                for imf_index in range(max_imft):
                    current_imf_sample_index = np.random.choice(indices_lists[class_index])
                    current_imf_sample = train_X[current_imf_sample_index]
                    for channel in range(num_channels):
                        channel_imf = emd.sift.sift(current_imf_sample[channel])
                        try:
                            augmented_sample[channel] += channel_imf[:, imf_index]
                        except IndexError:
                            pass
            else:
                class_index = index_to_class_dict[principal_sample_index]
                augmentation_sample_indexes = np.random.choice(indices_lists[class_index], shuffle_factor)
                augmented_sample = np.zeros((1, train_X.shape[1], train_X.shape[2]))
                for channel in range(train_X.shape[1]):
                    if np.random.random() > shuffle_channel_probability and np.random.random() > mirror_channel_probability:
                        augmented_sample[0, channel, :] = train_X[principal_sample_index, channel, :]
                    elif np.random.random() < shuffle_channel_probability and np.random.random() > mirror_channel_probability:
                        augmented_sample[0, channel, :] = train_X[np.random.choice(augmentation_sample_indexes),
                                                          channel, :]
                    elif np.random.random() > shuffle_channel_probability and np.random.random() < mirror_channel_probability:
                        reverted_channel_sample = train_X[principal_sample_index, channel, ::-1]
                        augmented_sample[0, channel, :] = reverted_channel_sample
                    else:
                        reverted_channel_sample = train_X[np.random.choice(augmentation_sample_indexes), channel, ::-1]
                        augmented_sample[0, channel, :] = reverted_channel_sample

            if np.random.random() < stft_noise_sample_probability:
                augmented_sample = add_noise_on_stft_sample_data(augmented_sample, fs=BOARD_SAMPLING_RATE,
                                                                 window_size=stft_window_size,
                                                                 gaussian_noise_std=gaussian_noise_stft_std)
            batch_train_X = np.append(batch_train_X, augmented_sample, axis=0)
            batch_train_y = np.append(batch_train_y, class_index)

            if len(batch_train_X) == batch_size:
                gaussian_noise_matrix = np.random.normal(loc=0, scale=gaussian_noise_std, size=batch_train_X.shape)
                batch_train_X = batch_train_X + gaussian_noise_matrix
                yield batch_train_X, batch_train_y
                batch_train_X = np.empty((0, train_X.shape[1], train_X.shape[2]))
                batch_train_y = np.empty(0)

        if len(batch_train_X) > 0:  # prevent yielding empty batch when len(train_X) % batch_size == 0
            gaussian_noise_matrix = np.random.normal(loc=0, scale=gaussian_noise_std, size=batch_train_X.shape)
            batch_train_X = batch_train_X + gaussian_noise_matrix
            yield batch_train_X, batch_train_y  # yield last batch of the epoch


class TrainSequenceWithAug(Sequence):
    """
        Keras Sequence with same funtionalities of 'train_generator_with_aug'
    """

    def __init__(self, train_X: np.ndarray, train_y: np.ndarray, batch_size: int,
                 shuffle_channel_probability=0., mirror_channel_probability=0.,
                 stft_noise_sample_probability=0., shuffle_factor=0, gaussian_noise_std=0.,
                 gaussian_noise_stft_std=0., stft_window_size=20, emd_sample_probability=0, max_imft=6):
        self.max_imft = max_imft
        self.emd_sample_probability = emd_sample_probability
        self.stft_window_size = stft_window_size
        self.gaussian_noise_stft_std = gaussian_noise_stft_std
        self.gaussian_noise_std = gaussian_noise_std
        self.shuffle_factor = shuffle_factor
        self.stft_noise_sample_probability = stft_noise_sample_probability
        self.mirror_channel_probability = mirror_channel_probability
        self.shuffle_channel_probability = shuffle_channel_probability
        self.batch_size = batch_size
        self.train_y = train_y
        self.train_X = train_X
        epoch_indexes = np.arange(len(self.train_X))
        np.random.shuffle(epoch_indexes)
        self.train_X = self.train_X[epoch_indexes]
        self.train_y = self.train_y[epoch_indexes]

        self.num_channels = train_X.shape[1]
        self.num_classes = int(np.max(train_y) + 1)
        self.indices_lists = [np.where(train_y == i)[0] for i in range(self.num_classes)]

    def __len__(self):
        return math.ceil(len(self.train_X) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.train_X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size]

        for sample_idx, sample in enumerate(batch_x):
            class_index = int(batch_y[sample_idx])
            if np.random.random() < self.emd_sample_probability:
                batch_x[sample_idx] = np.zeros_like(batch_x[sample_idx])
                for imf_index in range(self.max_imft):
                    current_imf_sample_index = np.random.choice(self.indices_lists[class_index])
                    current_imf_sample = self.train_X[current_imf_sample_index]
                    for channel in range(self.num_channels):
                        channel_imf = emd.sift.sift(current_imf_sample[channel])
                        try:
                            batch_x[sample_idx, channel, :] += channel_imf[:, imf_index]
                        except IndexError:
                            pass
            else:
                augmentation_sample_indexes = np.random.choice(self.indices_lists[class_index], self.shuffle_factor)
                # augmented_sample = np.zeros((1, self.train_X.shape[1], self.train_X.shape[2]))
                for channel in range(self.train_X.shape[1]):
                    if np.random.random() > self.shuffle_channel_probability and np.random.random() > self.mirror_channel_probability:
                        pass
                    elif np.random.random() < self.shuffle_channel_probability and np.random.random() > self.mirror_channel_probability:
                        batch_x[sample_idx, channel, :] = self.train_X[np.random.choice(augmentation_sample_indexes),
                                                          channel, :]
                    elif np.random.random() > self.shuffle_channel_probability and np.random.random() < self.mirror_channel_probability:
                        reverted_channel_sample = batch_x[sample_idx, channel, ::-1]
                        batch_x[sample_idx, channel, :] = reverted_channel_sample
                    else:
                        reverted_channel_sample = self.train_X[np.random.choice(augmentation_sample_indexes), channel,
                                                  ::-1]
                        batch_x[sample_idx, channel, :] = reverted_channel_sample

            if np.random.random() < self.stft_noise_sample_probability:
                batch_x[sample_idx] = add_noise_on_stft_sample_data(sample, fs=BOARD_SAMPLING_RATE,
                                                                    window_size=self.stft_window_size,
                                                                    gaussian_noise_std=self.gaussian_noise_stft_std)
        gaussian_noise_matrix = np.random.normal(loc=0, scale=self.gaussian_noise_std, size=batch_x.shape)
        batch_x = batch_x + gaussian_noise_matrix

        return batch_x, batch_y

    def on_epoch_end(self):
        epoch_indexes = np.arange(len(self.train_X))
        np.random.shuffle(epoch_indexes)
        self.train_X = self.train_X[epoch_indexes]
        self.train_y = self.train_y[epoch_indexes]
        self.indices_lists = [np.where(self.train_y == i)[0] for i in range(self.num_classes)]
