import json
import os
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, \
    SpatialDropout2D, SeparableConv2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Concatenate, Reshape, LeakyReLU, UpSampling2D, Embedding, \
    Multiply, LayerNormalization

from custom_callbacks import GANMonitor
from src.dataset_tools import load_all_raw_data, preprocess_raw_eeg

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # shuts down GPU

# Comment the following lines in order to occupy all GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def build_discriminator(Chans=8, Samples=250,
                        dropoutRate=0.5, kernLength=125, F1=12,
                        D=2, F2=24, norm_rate=0.25, dropoutType='Dropout', use_sigmoid_activation=True):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 2))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(2, Chans, Samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = LeakyReLU(alpha=0.2)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = LeakyReLU(alpha=0.2)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    if use_sigmoid_activation:
        output = Dense(1, name='dense', kernel_constraint=max_norm(norm_rate), activation='sigmoid')(flatten)
    else:
        output = Dense(1, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)

    return Model(inputs=input1, outputs=output, name="EEGNet-discriminator")


def build_cgan_discriminator(img_shape, num_classes=3, use_sigmoid_activation=True):
    # Input image
    img = Input(shape=img_shape)

    # Label for the input image
    label = Input(shape=(1,), dtype='int32')

    # Label embedding:
    # ----------------
    # Turns labels into dense vectors of size z_dim
    # Produces 3D tensor with shape (batch_size, 1, 28*28*1)
    label_embedding = Embedding(num_classes,
                                np.prod(img_shape),
                                input_length=1)(label)

    # Flatten the embedding 3D tensor into 2D tensor with shape (batch_size, 28*28*1)
    label_embedding = Flatten()(label_embedding)

    # Reshape label embeddings to have same dimensions as input images
    label_embedding = Reshape(img_shape)(label_embedding)

    # Concatenate images with their label embeddings
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(use_sigmoid_activation=use_sigmoid_activation)

    # Classify the image-label pair
    classification = discriminator(concatenated)

    return Model([img, label], classification)


def build_generator(z_dim):
    model = Sequential()

    model.add(Dense(8 * 8 * 2, input_shape=(z_dim,)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Reshape((8, 8, 2)))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(2, kernel_size=(1, 8), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2,
                        depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(4, kernel_size=(1, 16), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(8, kernel_size=(1, 32), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(8, kernel_size=(1, 64), padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))
    model.add(DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=1, depthwise_constraint=max_norm(1.),
                              padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))

    model.add(SeparableConv2D(4, kernel_size=(1, 4)))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))
    model.add(DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=1, depthwise_constraint=max_norm(1.),
                              padding='same'))
    model.add(BatchNormalization(axis=2))
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(1, kernel_size=(1, 125), padding='same'))
    model.add(LayerNormalization(axis=2))

    return model


def build_cgan_generator(z_dim, num_classes=3):
    # Random noise vector z
    z = Input(shape=(z_dim,))

    # Conditioning label: integer 0-9 specifying the number G should generate
    label = Input(shape=(1,), dtype='int32')

    # Label embedding:
    # ----------------
    # Turns labels into dense vectors of size z_dim
    # Produces 3D tensor with shape (batch_size, 1, z_dim)
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)

    # Flatten the embedding 3D tensor into 2D tensor with shape (batch_size, z_dim)
    label_embedding = Flatten()(label_embedding)

    # Element-wise product of the vectors z and the label embeddings
    joined_representation = Multiply()([z, label_embedding])

    generator = build_generator(z_dim)

    # Generate image for the given label
    conditioned_img = generator(joined_representation)

    return Model([z, label], conditioned_img)


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        real_images, data_labels = data
        real_images = tf.expand_dims(real_images, axis=-1)
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator([random_latent_vectors, data_labels])

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        gaussian_noise = tf.random.normal(shape=tf.shape(combined_images), stddev=1e-3)
        combined_images = combined_images + gaussian_noise

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1)) * 0.9], axis=0  # One sided label smoothing
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        labels = tf.clip_by_value(labels, clip_value_min=0, clip_value_max=1)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, tf.concat([data_labels, data_labels], axis=0)])
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_images = self.generator([random_latent_vectors, data_labels])
            predictions = self.discriminator([generated_images, data_labels])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, data_labels):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, data_labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, data_labels = data
        real_images = tf.expand_dims(real_images, axis=-1)

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, data_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, data_labels], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([real_images, data_labels], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, data_labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, data_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, data_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def wgan_discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def wgan_generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def fit_GAN(train_X, train_y, gan_hyperparameters_dict):
    latent_dim = gan_hyperparameters_dict['latent_dim']
    epochs = gan_hyperparameters_dict['epochs']
    batch_size = gan_hyperparameters_dict['batch_size']

    num_classes = int(np.max(train_y) + 1)
    img_shape = train_X[0, ..., None].shape  # to transform (8,250) shape into (8,250,1)

    generator = build_cgan_generator(latent_dim, num_classes=num_classes)
    discriminator = build_cgan_discriminator(img_shape=img_shape)

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_path: Path = Path(f"../GAN_{training_start}")
    model_path.mkdir(exist_ok=True, parents=True)
    callbacks_list = get_callback_lists(model_path, latent_dim)
    save_hyperparameters_dicts(gan_hyperparameters_dict, model_path)

    history = gan.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    plot_gan_losses(history, model_path)


def fit_WGAN(train_X, train_y, gan_hyperparameters_dict):
    latent_dim = gan_hyperparameters_dict['latent_dim']
    epochs = gan_hyperparameters_dict['epochs']
    discriminator_extra_steps = gan_hyperparameters_dict['wgan_discriminator_extra_steps']
    batch_size = gan_hyperparameters_dict['batch_size']

    num_classes = int(np.max(train_y) + 1)
    img_shape = train_X[0, ..., None].shape  # to transform (8,250) shape into (8,250,1)

    generator = build_cgan_generator(latent_dim, num_classes=num_classes)
    discriminator = build_cgan_discriminator(img_shape=img_shape, use_sigmoid_activation=False)

    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim,
                discriminator_extra_steps=discriminator_extra_steps)

    wgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
        g_loss_fn=wgan_generator_loss,
        d_loss_fn=wgan_discriminator_loss,
    )

    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_path: Path = Path(f"../WGAN_{training_start}")
    model_path.mkdir(exist_ok=True, parents=True)
    callbacks_list = get_callback_lists(model_path, latent_dim)
    save_hyperparameters_dicts(gan_hyperparameters_dict, model_path)

    history = wgan.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    plot_gan_losses(history, model_path)


def get_callback_lists(model_path, latent_dim=10):
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_path}' + '/saved-models-{epoch:06d}.h5',
            save_best_only=False
        ),
        keras.callbacks.CSVLogger(
            filename=f'{model_path}/my_model.csv',
            separator=',',
            append=True
        ),
        GANMonitor(
            model_path=model_path,
            latent_dim=latent_dim

        )
    ]
    return callbacks_list


def plot_gan_losses(history, model_path):
    plt.clf()
    d_loss = history.history['d_loss']
    g_loss = history.history['g_loss']
    epochs = range(1, len(d_loss) + 1)
    plt.plot(epochs, d_loss, 'g', label='Discriminator loss')
    plt.plot(epochs, g_loss, 'b', label='Generator loss')
    plt.title('Discriminator and Generator loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(f'{model_path}/generator-discriminator-loss')


def save_hyperparameters_dicts(gan_hyperparameters_dict, model_path):
    with open(model_path / "gan_parameters.json", 'w+') as file:
        json.dump(gan_hyperparameters_dict, file, sort_keys=True, indent=4)


def main():
    STARTING_DIR = Path("../chris_personal_dataset")
    SPLITTING_PERCENTAGE = namedtuple('SPLITTING_PERCENTAGE', ['train', 'val', 'test'])
    SPLITTING_PERCENTAGE.train, SPLITTING_PERCENTAGE.val, SPLITTING_PERCENTAGE.test = (70, 20, 10)
    raw_data_X, data_y = load_all_raw_data(starting_dir=STARTING_DIR)
    data_X, fft_data_X = preprocess_raw_eeg(raw_data_X, lowcut=8, highcut=45, coi3order=0)
    tmp_train_X, test_X, tmp_train_y, test_y = train_test_split(data_X, data_y,
                                                                test_size=SPLITTING_PERCENTAGE.test / 100,
                                                                random_state=42, stratify=data_y)
    actual_valid_split_fraction = SPLITTING_PERCENTAGE.val / (100 - SPLITTING_PERCENTAGE.test)
    train_X, val_X, train_y, val_y = train_test_split(tmp_train_X, tmp_train_y, test_size=actual_valid_split_fraction,
                                                      random_state=42, stratify=tmp_train_y)

    gan_hyperparameters_dict = {'latent_dim': 10,
                                'epochs': 30000,
                                'batch_size': 32,
                                'wgan_discriminator_extra_steps': 3}

    fit_GAN(train_X, train_y, gan_hyperparameters_dict)


if __name__ == '__main__':
    main()