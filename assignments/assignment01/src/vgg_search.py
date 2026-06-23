import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_index",
                        type=int,
                        default=42)
    return parser.parse_args()

class VggClassifier:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def load_data(self, data_dir):
        self.data = list(self.data_dir.glob('*.jpg'))
        self.filenames = [img.name for img in self.data]

    def model_setup(self):
        model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))
        return model

    def _extract_feature(self, data_path, model):
        """
        Extract features from image data using pretrained VGG model
        """
        # Define input image shape - remember we need to reshape
        input_shape = (224, 224, 3)
        # load image from file path
        img = load_img(data_path, target_size=(input_shape[0], 
                                            input_shape[1]))
        # convert to array
        img_array = img_to_array(img)
        # expand to fit dimensions
        expanded_img_array = np.expand_dims(img_array, axis=0)
        # preprocess image - see last week's notebook
        preprocessed_img = preprocess_input(expanded_img_array)
        # use the predict function to create feature representation
        # this works because the vgg model is loaded without the classification head
        features = model.predict(preprocessed_img, verbose=False)
        # flatten
        flattened_features = features.flatten()
        return flattened_features

    def extract_features(self, model):
        self.feature_list = [
            self._extract_feature(image, model)
            for image in tqdm(self.data, desc='Extracting features')
        ]

    def get_neighbors(self):
        print('getting nearest neighbors...')
        neighbors = NearestNeighbors(n_neighbors=10,
                                    algorithm='brute',
                                    metric='cosine').fit(self.feature_list)
        return neighbors

    def neighbor_distance(self, neighbors, target_index):
        distances, indices = neighbors.kneighbors([self.feature_list[target_index]])
        distance_list = []
        filename_list = []
        five_most_similar = []
        # This has to range up to 7 because target image is computed here
        for i in range(1,7):
            neighbor_idx = indices[0][i]
            distance_list.append(distances[0][i])
            filename_list.append(self.filenames[indices[0][i]])
            image = cv2.imread(str(self.data[neighbor_idx]))
            five_most_similar.append((self.filenames[indices[0][i]], distances[0][i], image))

        five_most_similar.sort(key=lambda x: x[1])
        arr = np.column_stack((filename_list, distance_list))
        target_image = cv2.imread(self.data[target_index])
        return five_most_similar, arr, target_image
        
    def plot_similar(self, five_most_similar, target_image):
        fig, axes = plt.subplots(1,6, figsize=(15,3))
        #Target image
        axes[0].imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("TARGET", fontsize=9, fontweight="bold")
        axes[0].axis("off")
        #similar images
        for ax, (filename, distance, image) in zip(axes, five_most_similar):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            ax.set_title(f'{distance:.3f}', fontsize=9)
            ax.set_xlabel(Path(filename).name, fontsize=7)
        plt.suptitle('Target vs. 5 most similar images (VGG - Cosine Similarity)')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/vgg_images.jpg')

    def save_distances(self, arr):
        filename = f'vgg_distances.csv'
        np.savetxt(
            f'{self.out_dir}/{filename}',
            arr,
            delimiter=',',
            header='Filename,Distance',
            fmt='%s'
        )
        print(f'File saved to {self.out_dir}/{filename}')

    def run(self, args):
        self.load_data(self.data_dir)
        model = self.model_setup()
        self.extract_features(model)
        neighbors = self.get_neighbors()
        similar, arr, target_image = self.neighbor_distance(neighbors, target_index=args.target_index)
        self.plot_similar(similar, target_image)
        self.save_distances(arr)

if __name__ == "__main__":
    args = get_args()
    root_dir = Path(__file__).parent.parent
    classifier = VggClassifier(
        data_dir=root_dir / "flowers",
        out_dir=root_dir / "out",
    )
    classifier.run(args)