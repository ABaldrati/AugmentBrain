import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
from dataset_tools import load_all_raw_data, preprocess_raw_eeg

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # shuts down GPU

# Comment the following lines in order to occupy all GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def build_discriminator(Chans=8, Samples=250,
                        dropoutRate=0.5, kernLength=125, F1=12,
                        D=2, F2=24, norm_rate=0.25, dropoutType='Dropout', use_sigmoid_activation=True):
    """
        GAN discriminator based on EEGNet
        http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
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
                    input_shape=(Chans, Samples, 2),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU(alpha=0.2)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = LeakyReLU(alpha=0.2)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    if use_sigmoid_activation:
        output = Dense(1, name='output', kernel_constraint=max_norm(norm_rate), activation='sigmoid')(flatten)
    else:
        output = Dense(1, name='output', kernel_constraint=max_norm(norm_rate))(flatten)

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
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2,
                        depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(4, kernel_size=(1, 16), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(8, kernel_size=(1, 32), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(
        DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(UpSampling2D(size=(1, 2), interpolation='bilinear'))
    model.add(SeparableConv2D(8, kernel_size=(1, 64), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=1, depthwise_constraint=max_norm(1.),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(4, kernel_size=(1, 4)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(DepthwiseConv2D((16, 1), use_bias=False, depth_multiplier=1, depthwise_constraint=max_norm(1.),
                              padding='same'))
    model.add(BatchNormalization())
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
    """
        GAN code heavily based on https://keras.io/examples/generative/dcgan_overriding_train_step/
    """

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
    """
        WGAN code heavily based on https://keras.io/examples/generative/wgan_gp/
    """

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
    discriminator = build_cgan_discriminator(img_shape=img_shape, use_sigmoid_activation=True)

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
    models_path = model_path / 'models'
    models_path.mkdir(exist_ok=True)
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{models_path}' + '/saved-models-{epoch:06d}.h5',
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
    raw_data_X, data_y, label_mapping = load_all_raw_data(starting_dir=STARTING_DIR)
    data_X, fft_data_X = preprocess_raw_eeg(raw_data_X, lowcut=8, highcut=45, coi3order=0)

    gan_hyperparameters_dict = {'latent_dim': 50,
                                'epochs': 30000,
                                'batch_size': 32,
                                'wgan_discriminator_extra_steps': 5,
                                'label_mapping': label_mapping}

    fit_WGAN(data_X, data_y, gan_hyperparameters_dict)
    # fit_GAN(train_X, train_y, gan_hyperparameters_dict)


def generate_synthetic_data(model_folder: Path, samples_to_generate: int, attempts: int, label: int, latent_dim: int,
                            initial_epoch=20000, epoch_step=50, data_shape=(8, 250, 1), num_classes=3):
    if type(model_folder) is str:
        model_folder = Path(model_folder)
    generated_samples = np.zeros((samples_to_generate,) + data_shape)
    for i in range(samples_to_generate):
        generator = build_cgan_generator(latent_dim)
        discriminator = build_cgan_discriminator(data_shape, num_classes, False)
        model = WGAN(discriminator, generator, latent_dim, 5)
        model.built = True
        model.load_weights(str(model_folder / Path(f'saved-models-{initial_epoch:06d}.h5')))

        noise = np.random.normal(loc=0, scale=1, size=(attempts, latent_dim))
        labels = np.ones((attempts, 1)) * label

        generated_sample = model.generator.predict([noise, labels])
        predictions = model.discriminator.predict([generated_sample, labels])

        flattened_predictions = predictions.flatten()
        max_index = np.argmax(flattened_predictions)

        generated_samples[i] = generated_sample[max_index]
        initial_epoch += epoch_step

    return generated_samples


if __name__ == '__main__':
    main()
