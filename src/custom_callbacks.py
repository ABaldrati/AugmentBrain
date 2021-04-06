from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf
import numpy as np
from dataset_tools import visualize_save_data


class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, model_path: Path, frequency=100, num_classes=3, sample_per_class=2, latent_dim=10):
        self.sample_per_class = sample_per_class
        self.num_classes = num_classes
        self.frequency = frequency
        self.latent_dim = latent_dim
        self.num_img = num_classes * sample_per_class

        self.sample_folder: Path = model_path / "logged_sample"
        self.sample_folder.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        labels = np.array([[i // self.sample_per_class] for i in range(self.num_img)])
        generated_samples = self.model.generator.predict([random_latent_vectors, labels])

        for idx, (sample, label) in enumerate(zip(generated_samples, labels)):
            name = f"generated_sample_epoch_{epoch}_label_{label}_idx_{idx}"
            visualize_save_data(sample, title=name, visualize_data=False,
                                file_path=self.sample_folder / Path(f"{name}.png"), save_data=True)
