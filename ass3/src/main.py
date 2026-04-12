import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Input,
    Rescaling, RandomFlip, RandomRotation, RandomZoom, Dropout
)

class LegoClassifier:
    """End-to-end LEGO image classifier using two different CNNs."""
    def __init__(self, data_dir: Path, out_dir: Path, epochs: int = 10, batch_size: int = 32):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_names = []
        self.model = None

        os.makedirs(self.out_dir, exist_ok=True)

    # argparse
    def get_args():
        parser = argparse.ArgumentParser(
            prog="""Choosing data type between base and cropped images.
            Also choosing between type of classifier, straight LeNet or using VGG16 for feature extraction first.""")

        parser.add_argument("--model",
                            type=str,
                            default="both",
                            required=False,
                            help="choice of CNN between VGG, LeNet, or both. Defaults to both")

        parser.add_argument("--data",
                            type=str,
                            default="both",
                            required=False,
                            help="choice between base, cropped, or both types of data. Defaults to both")

        parser.add_argument("--epochs",
                            type=int,
                            default=10,
                            required=False,
                            help="no. of training epochs, defaults to 10")

        parser.add_argument("--batch_size",
                            type=int,
                            default=32,
                            required=False,
                            help="batch size, defaults to 32")

        args = parser.parse_args()
        return args

    # Data loading
    def _load_split(self, subset_dir: str, pad: bool = False) -> tuple:
        """
        Loads train/val splits from a subdirectory, then carves the val
        split into val + test (80/10/10 overall).
        """
        # these args are passed to all the data loading, so keeping them in a shared variable
        shared_kwargs = dict(
            labels="inferred",
            seed=42,
            validation_split=0.2,
        )
        # flag to switch between padding and cropping based on the dataset
        aspect_kwarg = dict(pad_to_aspect_ratio=True) if pad else dict(crop_to_aspect_ratio=True)

        train = tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.data_dir}/{subset_dir}", subset="training",
            **shared_kwargs, **aspect_kwarg
        )
        val_full = tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.data_dir}/{subset_dir}", subset="validation",
            **shared_kwargs, **aspect_kwarg
        )

        # Split the 20% val 50/50 into val / test (80/10/10 final)
        shuffled = val_full.shuffle(
            buffer_size=val_full.cardinality().numpy(),
            seed=42,
            reshuffle_each_iteration=False,
        )
        split_size = int(0.5 * val_full.cardinality().numpy())
        val  = shuffled.take(split_size)
        test = shuffled.skip(split_size)

        return train, val, test

    def load_data(self) -> None:
        """Loads data into relevant variables. Also returns a forced ordered class_names variable."""
        self.base_train, self.base_val, self.base_test = self._load_split("base", pad=False)
        self.cropped_train, self.cropped_val, self.cropped_test = self._load_split("cropped", pad=True)
        self.class_names = sorted([item.name for item in (self.data_dir / "cropped").glob("*") if item.is_dir()]) # forcing ordered classes

    # Model setup
    def build_lenet_model(self) -> None:
        # seed on data augmentation for reproducibility
        data_augmentation = Sequential([
            RandomFlip("horizontal_and_vertical", seed=42, input_shape=(256, 256, 3)),
            RandomRotation(0.1, seed=42),
            RandomZoom(0.1, seed=42),
        ])
        # actual classifier (based on the LeNet from class)
        model = Sequential([
            data_augmentation,# as this contains input shape no input layer should be required
            Rescaling(1./256), # standardizing layer
            Conv2D(32, (3, 3), padding="same", activation="relu"), # first conv layer with relu activation
            MaxPool2D(pool_size=(2, 2)), # pooling
            Conv2D(50, (5, 5), padding="same", activation="relu"), # second conv
            MaxPool2D(pool_size=(2, 2)), # pooling
            Dropout(0.1), # 10% dropout, llms typically use 5%
            Flatten(), # flattening before fully connected layer
            Dense(128, activation="relu"), # dense relu layer
            Dense(20, activation="softmax"), # softmax output layer
        ])
        # compiling the model
        model.compile(
            loss="sparse_categorical_crossentropy", # supports multiple outputs
            optimizer="adam", # adam over sgd
            metrics=["accuracy"],
        )

        return model

    def build_vgg_model(self):
        # vgg setup without classificiation top
        vgg = VGG16(weights="imagenet", # unsure if this needs to be specified
                    include_top=False,
                    pooling="avg",
                    input_shape=(256,256,3)) # forcing input size to fit with dataset_from_directory default
        # Data augmentation with set seed for reproducability
        data_augmentation = Sequential([
            RandomFlip("horizontal_and_vertical", # random flips both axes
                            seed=42,
                            input_shape=(256,
                                        256,
                                        3)),
            RandomRotation(0.1, seed=42), # 10% random rotation and zoom
            RandomZoom(0.1, seed=42),])
        # freezing layers to avoid training them
        for layer in vgg.layers:
            layer.trainable = False

        # add new classifier layers
        inputs = Input(shape=(256, 256, 3)) # unsure if this is needed here because data augment contains it but the structure requires it for now
        # data augmentation defined above
        x = data_augmentation(inputs)
        # preprocess as a layer in the model - do not pass mapped data
        x = preprocess_input(x)
        # adding the vgg layer on top of the augmentation layer
        x = vgg(x)
        # fully connected relu layer
        x = Dense(128, activation='relu')(x)
        # add batch normalization
        x = BatchNormalization()(x)
        # 10% dropout, llms typically use 5%
        x = Dropout(0.1)(x)
        # softmax output classifier
        outputs = Dense(20, activation='softmax')(x)
        # generic model object for compiling
        model = Model(inputs=inputs, outputs=outputs)

        # compiling before returning
        model.compile(loss="sparse_categorical_crossentropy", # supports multiple output options
                optimizer="adam", # doing adam instead of stochastic gradient descent
                metrics=["accuracy"]) # right now optimizing for accuracy

        return model

    # Training and evaluation
    def train(self, train_data, val_data) -> tf.keras.callbacks.History:
        """Fits the model and returns the history object."""
        if self.model is None:
            raise RuntimeError("Call build_lenet_model() or build_vgg_model() before train().")

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            shuffle=True,
            epochs=self.epochs,
            verbose=1,
        )
        return history

    def evaluate(self, test_data) -> None:
        self.model.evaluate(test_data)

    def predict(self, test_data) -> tuple[np.ndarray, np.ndarray]:
        """Returns (y_true, y_pred) arrays for a dataset."""
        y_true, y_pred = [], []
        for images, labels in test_data:
            preds = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())
        return np.array(y_true), np.array(y_pred)

    # Reporting
    def save_report(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
        """writes a classification report to the out directory."""
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)
        path = self.out_dir / f"classification_report_{name}.txt"
        path.write_text(f"#### Classification report – {name} ####\n\n{report}")

    def save_history_plot(self, history: tf.keras.callbacks.History, name: str) -> None:
        """Saves a loss + accuracy training curve figure."""
        epochs = range(self.epochs)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 6))

        ax_loss.plot(epochs, history.history["loss"], label="train_loss")
        ax_loss.plot(epochs, history.history["val_loss"], label="val_loss", linestyle=":")
        ax_loss.set(title=f"Loss curve – {name}", xlabel="Epoch", ylabel="Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, history.history["accuracy"],     label="train_acc")
        ax_acc.plot(epochs, history.history["val_accuracy"], label="val_acc", linestyle=":")
        ax_acc.set(title=f"Accuracy curve – {name}", xlabel="Epoch", ylabel="Accuracy")
        ax_acc.legend()

        fig.tight_layout()
        fig.savefig(self.out_dir / f"training_values_{name}.png")
        plt.close(fig)

    # wrapper for running the classification
    def run(self, subset: str = "cropped", model: str = "lenet") -> None:
        """
        Convenience method that wires together the full pipeline.
        subset: "base" | "cropped" | "both"
        model: "lenet" | "vgg" | "both"
        """
        self.load_data()

        # logic for creating a model dictionary depending on arguments from argparse
        models = {}
        if model in ("lenet", "both"):
            models["lenet"] = self.build_lenet_model()
        if model in ("vgg", "both"):
            models["vgg"] = self.build_vgg_model()

        # same logic for choice of training data
        datasets = []
        if subset in ("base", "both"):
            datasets.append(("base", self.base_train, self.base_val, self.base_test))
        if subset in ("cropped", "both"):
            datasets.append(("cropped", self.cropped_train, self.cropped_val, self.cropped_test))

        # this is the core loop that allows for full implementation depending on choices 
        # of model and data above
        targets = [
            (model_name, model_obj, data_name, train, val, test)
            for model_name, model_obj in models.items()
            for data_name, train, val, test in datasets]

        for model_name, model_obj, data_name, train, val, test in targets:
            run_name = f'{model_name}_{data_name}'
            print(f"\n{"#"*30}\nTraining {model_name} on {data_name} images\n{"#"*30}")
            self.model = model_obj # extracting model object depending on model choice
            history = self.train(train, val)
            self.save_history_plot(history, run_name)
            self.evaluate(test) # eval on held out test data
            y_true, y_pred = self.predict(test) # predictions from model
            self.save_report(y_true, y_pred, run_name)

if __name__ == "__main__":
    args = LegoClassifier.get_args()
    root_dir = Path(__file__).parent.parent
    classifier = LegoClassifier(
        data_dir=root_dir / "data",
        out_dir=root_dir / "out",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    classifier.run(subset=args.data, model=args.model)