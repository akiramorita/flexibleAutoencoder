# Introduction
This program is flexible autoencoder in terms of CNN and MLP structure and parameters. It outcomes autoencoder and encoder weights.
# Autoencoder
## General descriptions and applications
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name. Several variants exist to the basic model, with the aim of forcing the learned representations of the input to assume useful properties. Examples are the regularized autoencoders (Sparse, Denoising and Contractive autoencoders), proven effective in learning representations for subsequent classification tasks, and Variational autoencoders, with their recent applications as generative models. Autoencoders are effectively used for solving many applied problems, from face recognition to acquiring the semantic meaning of words. Reference: Autoencoder - Wikipedia

## Structure
Autoencoder consists of encoder and decoder. Encoder converts high-dimensional inputs such as image data to relatively low-dimensional space: embedding. Then decoder reverse it to try to reproduce the original one with the same coding but reverse process.

In this program, autoencoder includes a sequence of Convolution Neural Networks (CNN) and multilayer perception (MLP): 

inputs -> CNN -> MLP -> embedding -> MLP -> CNN -> outputs

## Model training
Autoencoder model training is a supervised way: comparing inputs and outputs. The inputs and outputs are supposed to be the same. Its loss function is binary_crossentropy to converge the model.

This program gives adjustments of CNN and MLP structure and parameters for performance improvement: ex. number of nodes, number of layers, CNN filter, pooling, kernel size, stride, and Dense parameters.

## Outcome
Autoencoder model and encoder model are the training outcome. The encoder model can be used for k-means clustering process to convert input images to low dimension embeddings.

# Image requirements
This program gives adjustability of image size, but two practical things need to be taken care of: CNN structure and computing power. Depending on CNN, acceptable image sizes are scattered. The reason for it should be referred to CNN textbook. 

The larger images, the more computing power are required. In using GPU, GPU memory size determines the acceptable neural network size for training. The larger images tend to need larger network for better performance. Thus the images should be limited to a key part to make them meaningful in terms of clustering. 

This program’s default image size is 256 x 256.

# Packages
* tensorflow 1.13.1
* keras 2.2.4
* pandas 0.24.2
* numpy 1.16.4
* cv2 4.1.0.25
* fire 0.2.1

# Setup and Execution
## Setup
    $git clone https://github.com/akiramorita/flexibleAutoencoder.git

    $cd ./flexibleAutoencoder

* Store image (256x256) files under data/ by following examples;

    * Autoencoder/data/train/class_0/*.png
    * Autoencoder/data/valid/class_0/*.png

## Execution
    $python -m src.autoencoder run

# Internal step descriptions
## Adjust training parameter

You may consider training parameter adjustments. Major parameters are in the following.
    
* imageShape (image shape)
* train_epochs (number of train epoch)
* batch_size (batch size)
* filters (CNN filter structure)
* pooling (CNN pooling structure)
* kernel_size (CNN kernel structure)
* strides (CNN stride structure)
* activation (CNN and MPL activation) 
* last_activation (CNN last layer activation)
* optimizer (optimizer)
* dropRate (CNN and MPL dropout rate)
* dims (MPL node and layer structure)
* patience (epoch count for early stopping)
* period (epoch frequency for weight saving)
* AutoencoderMLP_switch (MPL on/off)

### Execution example with adjustments
	$ python -m src.autoencoder --train_epochs=100 --filters=[128,128,128] run

## Execute model training

* Adjust the length of pooling, kernel_size and strides according to filter length
* Construct CNN and MPL structure in the following section according to filters, pooling, kernel_size, strides, and dims
* Prepare generator to feed data to model: no need to load all of data once
* Prepare sample data for performance evaluation
* Construct autoencoder for sample data
* Construct autoencoder for training
* Train autoencoder
* Save model
* Output loss, accuracy, and before-and-after comparison images

# Review model performance
* Autoencoder/logs/aescore.csv is available to compare loss and accuracy by model folders
* Autoencoder model (ae_weights.h5), encoder model (en_weights.h5), and before-and-after comparison images are available in model folders under Autoencoder/logs/ 

