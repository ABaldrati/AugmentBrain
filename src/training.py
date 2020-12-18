from dataset_tools import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import cris_net, res_net, TA_CSPNN, EEGNet

from sklearn.model_selection import KFold, cross_val_score
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import time
import os
from pathlib import Path

import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # shuts down GPU


def fit_model(model: keras.models.Model, epochs, train_X, train_y, validation_X, validation_y, batch_size=1):
    model_name = model.name
    Path.mkdir(Path(f"../{model_name}"), exist_ok=True)

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=1500,
            mode='auto',
            #   min_delta=-0.0001
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f'../{model_name}'+'/saved-models-{epoch:06d}-{val_loss:.5f}.h5',
            monitor='val_loss',
            save_best_only=False
        ),
        keras.callbacks.CSVLogger(
            filename=f'../{model_name}/my_model.csv',
            separator=',',
            append=True
        ),
    ]
    history = model.fit(
        x=train_X,
        y=train_y,
        validation_data=(validation_X, validation_y),
        epochs=epochs,
        callbacks=callbacks_list,
        batch_size=batch_size
    )

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../{model_name}/training-validation-loss')

    plt.clf()
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(f'../{model_name}/training-validation-accuracy')


if __name__ == '__main__':
    split_data(shuffle=True, division_factor=0, coupling=False)

    # loading personal_dataset
    tmp_train_X, train_y = load_data(starting_dir="../training_data", shuffle=True, balance=True)
    tmp_validation_X, validation_y = load_data(starting_dir="../validation_data", shuffle=True, balance=True)

    # cleaning the raw personal_dataset
    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X, lowcut=8, highcut=45, coi3order=0)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X, lowcut=8, highcut=45, coi3order=0)

    # reshaping
    train_X = np.expand_dims(train_X, -1)
    validation_X = np.expand_dims(validation_X, -1)

    model = EEGNet(nb_classes=len(ACTIONS))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=5e-4),
                  metrics=['accuracy'])

    batch_size = 16
    epochs = 1000
    fit_model(model, epochs, train_X, train_y, validation_X, validation_y, batch_size)
