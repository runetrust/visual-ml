import argparse
from pathlib import Path
from class_logistic import LogisticClassifier
from class_nn import NeuralNetClassifier

def get_args():
    parser = argparse.ArgumentParser(
        prog="""Shared hyperparameters. Prefix nn for NeuralNet args,
        log for Logistic Regression"""
    )
    
    #Neural net args
    parser.add_argument("--nn_hidden_layers", 
                        type=tuple,
                        default=(64, 10))
    parser.add_argument("--nn_learning_rate",
                        type=str,
                        default="adaptive")
    parser.add_argument("--nn_solver",
                        type=str,
                        default="adam")
    parser.add_argument("--nn_max_iter",
                        type=int,
                        default=100)
    parser.add_argument("--nn_early_stopping",
                        action="store_true")
    #Logistic args-
    parser.add_argument("--log_tolerance",
                        type=float,
                        default=0.01)
    parser.add_argument("--log_max_iter",
                        type=int,
                        default=100)
    parser.add_argument("--log_solver",
                        type=str,
                        default="saga")
    #flags for both
    parser.add_argument("--verbose",
                        action="store_true")
    return parser.parse_args()

def main():
    args=get_args()

    root_dir = Path(__file__).parent.parent

    #neuralnet classification
    nn_classifier = NeuralNetClassifier(
        out_dir = root_dir / "out"
    )
    nn_classifier.runner(args)
    #logistic regression classification
    log_classifier = LogisticClassifier(
        out_dir = root_dir / "out"
    )
    log_classifier.runner(args)

if __name__ == "__main__":
    main()