import argparse
from pathlib import Path
from opencv_search import OpenCvClassifier
from vgg_search import VggClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_index",
                        type=int,
                        default=42,
                        help="Index of target image to compute neighbors for, defaults to 42")
    return parser.parse_args()

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    args = get_args()
    opencv_classifier = OpenCvClassifier(
        data_dir = root_dir / "flowers",
        out_dir = root_dir / "out"
    )
    opencv_classifier.runner(args)

    vgg_classifier = VggClassifier(
        data_dir=root_dir / "flowers",
        out_dir=root_dir / "out",
    )
    vgg_classifier.run(args)