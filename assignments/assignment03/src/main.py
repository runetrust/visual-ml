import argparse
from pathlib import Path
from lego_classification import LegoClassifier

def get_args():
    parser = argparse.ArgumentParser(
        prog="""Choosing data type between base and cropped images.
        Also choosing between type of classifier, straight LeNet or using VGG16 for feature extraction first.""")

    parser.add_argument("--model",
                        type=str,
                        default="both",
                        required=False,
                        help="choice of CNN between 'vgg', 'lenet', or both. Defaults to 'both'")

    parser.add_argument("--data",
                        type=str,
                        default="both",
                        required=False,
                        help="choice between 'base', 'cropped', or 'both' types of data. Defaults to 'both'")

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
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    root_dir = Path(__file__).parent.parent
    classifier = LegoClassifier(
        data_dir=root_dir / "data",
        out_dir=root_dir / "out",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    classifier.run(subset=args.data, model=args.model)