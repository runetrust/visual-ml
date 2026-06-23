import os
import cv2
import numpy as np
import glob
import tqdm
from pathlib import Path
from imutils import jimshow
from imutils import jimshow_channel
import matplotlib.pyplot as plt

class OpenCvClassifier:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def load_data(self):
        files = list(self.data_dir.glob('*.jpg'))
        self.filenames = [img.name for img in files]
        return files

    def process_data(self, files):
        images = [cv2.imread(image) for image in files]
        return images

    def select_target_img(self, images, target_index=None):
        '''
        Accepts target index, if not supplied then it picks a random number
        '''
        if target_index is not None:
            target_image = images[target_index]
        else:
            target_image = images[np.random.randint(len(images))]

        cv2.imwrite(os.path.join(self.out_dir, f"opencv_target_image.jpg"), target_image)

        target_hist = cv2.calcHist(target_image,
            [0,1,2], # 3 channels
            None, # mask
            [256,256,256], # histogram size (how many values of color?)
            [0,256, 0,256, 0,256]) #ranges
        target_hist = cv2.normalize(target_hist, target_hist, 0, 1.0, cv2.NORM_MINMAX) # Norm
        return target_hist

    def compute_distances(self, images, target_hist):
        hist_distances = []
        for i, image in tqdm.tqdm(enumerate(images),desc="Processing histograms"):
            hist = cv2.calcHist([image],
                [0,1,2], # 3 channels
                None, # mask
                [256,256,256], # histogram size (how many values of color?)
                [0,256, 0,256, 0,256]) #ranges
            hist_norm = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX) # Normalizing
            distance_metric = cv2.compareHist(target_hist,hist_norm, cv2.HISTCMP_CHISQR) # Computing dist
            hist_distances.append((self.filenames[i], distance_metric, image))

        hist_distances.sort(key=lambda x: x[1])
        arr = np.column_stack(([x[0] for x in hist_distances], [x[1] for x in hist_distances]))
        five_most_similar = hist_distances[1:6]# Only keeping most similar in new array (1 index because most similar is the target)
        return five_most_similar, arr

    def plot_similar(self, five_most_similar):
        fig, axes = plt.subplots(1,5, figsize=(15,3))
        for ax, (filename, distance, image) in zip(axes, five_most_similar):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            ax.set_title(f'{distance:.3f}', fontsize=9)
            ax.set_xlabel(Path(filename).name, fontsize=7)
        plt.suptitle('5 most similar images (OpenCV - Chi Squared)')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/opencv_images.jpg')

    def save_file(self, arr):
        np.savetxt(
            f'{self.out_dir}/opencv_distances.csv',
            arr,
            delimiter=',',
            header='File,Distance',
            fmt="%s"#forces mix of numbers and strings
        )
        print(f'File saved to {self.out_dir}/opencv_distances.csv')

    def runner(self, args):
        data = self.load_data()
        images = self.process_data(data)
        target_hist = self.select_target_img(images, target_index=args.target_index)#not selecting any target, using random number
        similar, arr = self.compute_distances(images, target_hist)
        self.plot_similar(similar)
        self.save_file(arr)

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    classifier = OpenCvClassifier(
        data_dir = root_dir / "flowers",
        out_dir = root_dir / "out"
    )
    classifier.runner()