# File heavily based on https://github.com/CrisSherban/BrainPad

from brainflow import DataFilter, FilterTypes, AggOperations
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy.fft import fft
from tqdm import tqdm

import numpy as np
import os

ACTIONS = ["feet", "none", "hands"]


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

    if not os.path.exists("../training_data") and not os.path.exists("../validation_data") \
            and not os.path.exists("../untouched_data"):

        # creating directories

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
                data[k, i, :] = (data[k, i, :] - mean) / std

    return data


def visualize_data(data, file_name, title, length):
    # takes a look at the personal_dataset
    for i in range(len(data[0])):
        plt.plot(np.arange(len(data[0][i])), data[0][i].reshape(length))

    plt.title(title)
    plt.savefig(file_name + ".png")
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
    # visualize_data(data,
    #                file_name="../pictures/before",
    #                title="RAW EEGs",
    #                length=len(data[0, 0]))

    data = standardize(data)

    # visualize_data(data,
    #                file_name="../pictures/after_std",
    #                title="After Standardization",
    #                length=len(data[0, 0]))

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

    # visualize_data(data,
    #                file_name="../pictures/after_bandpass",
    #                title=f'After bandpass from {lowcut}Hz to {highcut}Hz',
    #                length=len(data[0, 0]))
    # visualize_data(fft_data,
    #                file_name="../pictures/ffts",
    #                title="FFTs",
    #                length=len(fft_data[0, 0]))

    return data, fft_data


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


if __name__ == '__main__':
    split_data(shuffle=False, division_factor=0, coupling=False)

    # loading personal_dataset
    tmp_train_X, train_y = load_data(starting_dir="../training_data", shuffle=False, balance=True)
    tmp_validation_X, validation_y = load_data(starting_dir="../validation_data", shuffle=False, balance=True)

    # cleaning the raw personal_dataset
    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X, lowcut=8, highcut=45, coi3order=0)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X, lowcut=8, highcut=45, coi3order=0)

    check_duplicate(train_X, validation_X)
