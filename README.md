# AugmentBrain

![](logo/logo_large.png "Logo")


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Images](#images)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)


## About The Project
Brain computer interfaces provides a new communication bridge between human minds and devices, however the ability to control such devices with our minds largely depends on the accurate classification and identification of non-invasive EEG signals. For this reason recent advances in deep learning have helped the progress in such field with convolutional neural networks that are becoming the new cutting edges tools to tackle the problem of EEG recognition. In order to successfully train a convolutional neural network a large amount of data are needed and due to the strict requirements for subjects and experimental environments, it is difficult to collect large-scale and high-quality EEG data.
Based on this, in ```AugmentBrain``` we investigate the performance of different data augmentation methods for the classification of Motor Imagery (MI) data using a Convolutional Neural Network tailored for EEG named EEGNet.

All the work is based on [Serban Cristian Tudosie](https://github.com/CrisSherban) [BrainPad](https://github.com/CrisSherban/BrainPad) repository.


### Built With
**Software**:
* [Python](https://www.python.org/)
* [BrainFlow](https://brainflow.org/)
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [ScyPy](https://www.scipy.org/)

**Hardware**:
* [OpenBCI](https://shop.openbci.com/collections/frontpage)




## Getting Started

To get a local copy up and running follow these simple steps.

### Installation
 
1. Clone the repo
```sh
git clone https://gitlab.com/ABaldrati/hci-project
```
2. Install Python dependencies

## Usage
Here's a brief description of each and every file in the repo:

* ```training.py```: Model training
* ```dataset_tools.py```: dataset loading utils and preprocessing
* ```GAN.py```: GAN training
* ```neural_nets.py```: neural nets definitions
* ```custom_callbacks.py```: keras custom callbacks which is useful in model and GAN training
* ```acquire_eeg.py```: new EEG data acquisition

Obviously in order to acquire new EEG data OpenBCI Hardware is required.


## Authors

* [**Alberto Baldrati**](https://github.com/ABaldrati)

Based on [**Serban Cristian Tudosie**](https://github.com/CrisSherban) work.


## Acknowledgments
Human Computer Interaction © Course held by Professor [Andrew David Bagdanov](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
