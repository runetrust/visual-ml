import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

class LogisticClassifier:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def get_args():
        parser = argparse.ArgumentParser(
            prog="Logistic regression hyperparameters"
        )
        parser.add_argument("--log_tolerance",
                            "-tol",
                            type=float,
                            default=0.1,
                            required=False)
        parser.add_argument("--log_solver",
                            type=str,
                            default="saga",
                            required=False)
        parser.add_argument("--log_max_iter",
                            type=int,
                            default=100,
                            required=False)
        parser.add_argument("--verbose",
                            action="store_true")
        args = parser.parse_args()
        return args

    def load_data(self):
        """
        Loads and preprocesses data to a format ready for classification
        """
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Hardcoding labels as they are not present in the data
        self.labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"]
        
        # Grayscaling
        X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
        X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

        # Scaling
        X_train = X_train/255
        X_test = X_test/255

        # Reshaping
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))

        return X_train, y_train, X_test, y_test

    def classify_data(self, X_train,y_train,X_test, y_test, tol, verbose, max_iter, solver):
        clf = LogisticRegression(tol=tol,
                            verbose=verbose,
                            max_iter=max_iter,
                            solver=solver).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=self.labels)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{self.out_dir}/report_lr.txt'
        with open(file_name, "w") as f:
            f.write("#### Classification report LogisticRegression ####\n")
            f.write(report)

        print(f'\n\n#### Classification report saved to {file_name} ####\n\n')
        return y_pred

    def runner(self, args):
        X_train, y_train, X_test, y_test = self.load_data()
        print("\n\n#### Classifying data using Logistic Regression ####\n\n")
        self.classify_data(X_train, y_train, X_test, y_test, solver=args.log_solver,
        tol=args.log_tolerance, max_iter=args.log_max_iter, verbose=args.verbose)

if __name__ == "__main__":
    #this allows me to also just run the two classes seperately without wrapping them in the main.py
    args = LogisticClassifier.get_args()
    root_dir = Path(__file__).parent.parent
    classifier = LogisticClassifier(
        out_dir = root_dir / "out"
    )
    classifier.runner(args)