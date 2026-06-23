# Classification benchmarks with Logistic Regression and Neural Networks
This repo uses two different types of classification structures in order to classify images from the `Cifar10` dataset.
The first implementation is a simple Logistic Regression, and the second is a Neural Network.

# Data
The data for this is the `Cifar10` dataset (can be downloaded from [Toronto University](https://www.cs.toronto.edu/%7Ekriz/cifar.html)) which contains 60,000 images total with a train split of 50,000, and a test split of 10,000. The data is split into 10 classes, with 6,000 images in each.

The data is available via `Keras` Datasets, and is automatically downloaded when running the code.

# Repo Structure
```
out/
  └── report_lr.txt # Logistic Regression Classification Report
  └── report_nn.txt # Neural Network Classification Report
  └── nn_loss.jpg # Neural Network loss curve
src/
  └── main.py # Convenience wrapper which runs both classification types
  └── class_logistic.py # The Logistic Regression classifier
  └── class_nn.py # The Neural Network Classifier

setup.sh
requirements.txt
README.md
```

# Reproducing the Analysis
## 1. Setup
I have included a `setup.sh` script which sets up the virtual environment for running analysis.

This does:
```
python -m venv env
source ./env/bin/activate
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install -r requirements.txt
deactivate 
```
## 2. Activate Environment
Run
```
source env/bin/activate
```
## 3. Run Script
Do 
```
python src/main.py
```
The above script has sensible defaults, but argparse can be used to change almost all hyperparameters.
```
Shared hyperparameters. Prefix *nn* for NeuralNet args, *log* for Logistic Regression

options:
  --nn_hidden_layers NN_HIDDEN_LAYERS
  --nn_learning_rate NN_LEARNING_RATE
  --nn_solver NN_SOLVER
  --nn_max_iter NN_MAX_ITER
  --nn_early_stopping
  --log_tolerance LOG_TOLERANCE
  --log_max_iter LOG_MAX_ITER
  --log_solver LOG_SOLVER
  --verbose
```
Output is saved to the `out` folder.

## 4. Deactivate
```
deactivate
```

# Summary of Results
The first difference between the two implementations is that we see a ~ 0.10 increase in classification accuracy across all classes (f1-score). This means that simply changing the classification architecture affords a non-neglible improvement to the task.

### Logistic Regression
```
#### Classification report LogisticRegression ####
              precision    recall  f1-score   support

    airplane       0.34      0.37      0.35      1000
  automobile       0.35      0.37      0.36      1000
        bird       0.24      0.20      0.22      1000
         cat       0.20      0.15      0.18      1000
        deer       0.23      0.19      0.21      1000
         dog       0.28      0.29      0.29      1000
        frog       0.27      0.30      0.28      1000
       horse       0.28      0.28      0.28      1000
        ship       0.35      0.39      0.37      1000
       truck       0.38      0.43      0.41      1000

    accuracy                           0.30     10000
   macro avg       0.29      0.30      0.29     10000
weighted avg       0.29      0.30      0.29     10000
```
### Neural Network
```
#### Classification report Neural Network ####
              precision    recall  f1-score   support

    airplane       0.40      0.46      0.43      1000
  automobile       0.47      0.52      0.49      1000
        bird       0.32      0.25      0.28      1000
         cat       0.26      0.18      0.22      1000
        deer       0.32      0.34      0.33      1000
         dog       0.44      0.29      0.35      1000
        frog       0.36      0.53      0.43      1000
       horse       0.49      0.41      0.44      1000
        ship       0.44      0.55      0.49      1000
       truck       0.47      0.45      0.46      1000

    accuracy                           0.40     10000
   macro avg       0.40      0.40      0.39     10000
weighted avg       0.40      0.40      0.39     10000
```

What may be more interesting, however, is that we see a "flattening" of classification accuracy between classes. Certain classes are more difficult than others to classify (eg. cat vs. ship), and a Neural Network seems to make classification for the classes more similar than they are if using Logistic Regression.

# Limitations and Future Directions
Even though switching from a "simple" logistic regression to a more complex neural network does consequently improve performance and especially robustness between classes, classification is still pretty terrible.

Classification accuracy with an NN of around 40% is not great even though this is better than Logistic Regression around 30%. 

This essentially tells us that there is simply not enough information in the 50,000 samples to train a robust image classification model from scratch, so using a pretrained model for this classification task should prove a significant improvement.