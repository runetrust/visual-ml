# End to End LEGO Classifier Using Convolutional Neural Networks (CNNs).

## Description
This repo contains a main.py script for classification of LEGO images.

The script can run both a "Base" LeNet inspired CNN on the data, or it can use VGG16 as a feature extractor before passing to a CNN for classification.

Both CNN's utilize data augmentation to "increase" the amount of training data available to the model for learning patterns.

The data contains both "base", with the LEGO brick only taking up a small portion of the image, and "cropped" images which are cropped to only contain the brick - both have the brick against a white background with minimal noise.

## Running
### 1. 
Download the data from Kaggle here: https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition and unzip this into the data folder in this repo.

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
To run actual classification with the desired arguments. This takes a while.

## Outputs
The script saves to the out folder.

The classification reports indicate that using the pretraing VGG16 networks for a form of transfer learning raises classification accuracy a non-neglible amount from PLACEHOLDER to PLACEHOLDER.

Looking at the training plots, we also see that the PLACEHOLDER model seems to train more robustly, and for longer before plateauing and overfitting to the data. Both have in common that certain categories are harder to classify than others, seen by lower f1 scores for these categories.

## Future Improvements
The approach in this repo is far from perfect, even if it does constitute a viable approach to classifying the given images. 

Firstly, the base LeNet CNN has been created with "default" layers and is not as such tuned for optimal accuracy for the task at hand. The same goes for the classification layers following VGG16 in the VGG model. 

Secondly, no hyperparameter tuning was done. This means that the default values have not been selected from a place of maximizing accuracy to the task - this could potentially garner significant improvements to the classifiers. 

Thirdly, the data is split into train / val / test segments with a fixed seed. This seed does not return perfectly balanced datasets (seen from support in classification report) so a different seed could potentially return more balanced datasets which should theoretically improve model accuracy.