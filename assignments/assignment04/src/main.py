from pathlib import Path
from face_detection import FaceDetector

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    classifier = FaceDetector(
        data_dir = root_dir / "data",
        out_dir = root_dir / "out"
    )
    classifier.runner()