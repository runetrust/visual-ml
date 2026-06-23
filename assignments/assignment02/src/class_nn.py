import os
import cv2
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import cifar10

class NeuralNetClassifier:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def get_args():
        parser = argparse.ArgumentParser(
            prog="Neural Network hyperparameters"
        )
        parser.add_argument("--nn_hidden_layers",
                            "-hl",
                            type=tuple,
                            default=(64,10),
                            required=False)
        parser.add_argument("--nn_learning_rate",
                            "-lr",
                            type=str,
                            default="adaptive",
                            required=False)
        parser.add_argument("--nn_solver",
                            type=str,
                            default="adam",
                            required=False)
        parser.add_argument("--nn_max_iter",
                            type=int,
                            default=100,
                            required=False)
        parser.add_argument("--nn_early_stopping",
                            action="store_true",
                            required=False)
        parser.add_argument("--verbose",
                            action="store_true",
                            required=False)
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

    def classify_data(self, X_train,y_train,X_test, y_test, hidden_layers, solver,
                    learning_rate, early_stopping, verbose, max_iter):

        clf = MLPClassifier(random_state=42,
                            hidden_layer_sizes=hidden_layers,
                            solver=solver,
                            learning_rate=learning_rate,
                            early_stopping=early_stopping,
                            verbose=verbose,
                            max_iter=max_iter).fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=self.labels)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{self.out_dir}/report_nn.txt'
        with open(file_name, "w") as f:
            f.write("#### Classification report Neural Network ####\n")
            f.write(report)

        print(f'\n\n#### Classification report saved to {file_name} ####\n\n')
        # calling the plot function here to pass the timestamp instead of passing these around
        self.plot_loss_curve(clf, timestamp)
        return y_pred

    def plot_loss_curve(self, clf, timestamp):
        plt.figure(figsize=(8, 5))
        plt.plot(clf.loss_curve_, label="Training Loss")
        plt.title("Neural Network Training Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        file_name = f"{self.out_dir}/nn_loss.jpg"
        plt.savefig(file_name)
        plt.close()

    def runner(self, args):
        X_train, y_train, X_test, y_test = self.load_data()
        print("\n\n#### Classifying data using Neural Net ####\n\n")
        self.classify_data(X_train, y_train, X_test, y_test, hidden_layers=args.nn_hidden_layers, 
        solver=args.nn_solver, learning_rate=args.nn_learning_rate, early_stopping=args.nn_early_stopping,
        verbose=args.verbose, max_iter=args.nn_max_iter)

if __name__ == "__main__":
    args = NeuralNetClassifier.get_args()
    root_dir = Path(__file__).parent.parent
    classifier = NeuralNetClassifier(
        out_dir = root_dir / "out"
    )
    classifier.runner(args)