# End to End LEGO Classifier Using Convolutional Neural Networks (CNNs).

## Description
This repo contains a main.py script for classification of LEGO images.

The script can run both a "Base" LeNet inspired CNN on the data, or it can use VGG16 as a feature extractor before passing to a CNN for classification.

Both CNN's utilize data augmentation to "increase" the amount of training data available to the model for learning patterns.

The data contains both "base", with the LEGO brick only taking up a small portion of the image, and "cropped" images which are cropped to only contain the brick - both have the brick against a white background with minimal noise.

## Running
### 1. 
Download the data from Kaggle here: https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition and unzip this into the data folder in this repo. The data should be structured with a "base" folder and a "cropped" folder.

### 2.
Run setup script below in the terminal. This sets up a virtual environment and downloads the required packages.
```
setup.sh
```

### 3.
Run the command below in the terminal to activate environment.
```
source env/bin/activate
```

The main.py script defaults to running both classifiers on both data types for 10 epochs. Run the chunk below to check argparse arguments for changing this behavior.
```
python src/main.py -h
```

Finally do
```
python src/main.py
```
To run actual classification with the desired arguments, if any. This takes a while.

## Outputs
The script saves to the out folder.

The classification reports indicate that using the pretraing VGG16 network within a transfer learning paradigm raises classification accuracy on base images significantly, from an f1 accuracy of 0.53 in the LeNet implementation to 0.88 in the VGG implementation.

Looking at the training plots for base images, we see that the VGG model hits close to maximum accuracy after three epochs, which is also where training and validation loss begins to converge - indicating overfitting in the later epochs. If sticking to a strict early stopping regime, it could be argued that the model should stop training at this point, with a validation accuracy at ~0.85.

The LeNet model also shows signs of overfitting at the third epoch, but at a validation accuracy of only ~0.24. This tells us that implementing the pretrained VGG16 network has a very large impact on the accuracy of classification.

While the change in f1 accuracy in itself tells us that it is worth using a pretrained network in the architecture, this also becomes even clearer when looking at the details in the classification reports. The LeNet model struggles a lot in certain categories, with the lowest f1 accuracy at only 0.13, compared to the VGG lowest category being 0.71. This shows that not only does VGG increase accuracy on certain "easier" categories, it also raises the lowest bar significantly, flattening classification accuracy across the (lego)board.

## Future Improvements
The approach in this repo is far from perfect, even if it does constitute a viable approach to classifying the given images. 

Firstly, the base LeNet CNN has been created with "default" layers and is not as such tuned for optimal accuracy for the task at hand. The same goes for the classification layers following VGG16 in the VGG model. 

Secondly, no hyperparameter tuning was done. This means that the default values have not been selected from a place of maximizing accuracy to the task - this could potentially garner significant improvements to the classifiers. 

Thirdly, the data is split into train / val / test segments with a fixed seed. This seed does not return perfectly balanced datasets (seen from support in classification report) so a different seed could potentially return more balanced datasets which should theoretically improve model accuracy.
