import torch
import glob
import os
import tqdm
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def load_data(self, data_path):
        data_path = self.data_dir
        #Necessary to sort the input data because the function is funky
        data = sorted(Path(data_path).rglob("*.jpg"))
        return data
    
    def detect(self, data):
        '''
        Loops through data folder to detect faces using the MTCNN algorithm

        Newspaper name, date, decade, page number, and amount of faces are appended through a dictionary
        and saved to a csv file at the end.

        Certain images throw a truncation error within Pillow, but upon visual inspection
        I believe this actually represents corrupted images, and not a Pillow error. 
        For this reason I am catching these in a try-except block as they contain no information.
        '''
        mtcnn = MTCNN(keep_all=True)
        # This pattern gives me four groups for the different data types I need for the df
        pattern = r'(\w+)-((\d{4})-\d{2}-\d{2})-a-\w(\d{4})'

        face_list = []
        for image in tqdm.tqdm(data, desc="Detecting faces"):
            match = re.search(pattern, str(image))
            if match:
                newspaper = match.group(1)
                date = match.group(2)
                year = int(match.group(3))
                page = match.group(4)
            decade = ((year // 10) * 10)

            # Catching corrupted images and skipping
            try:
                img = Image.open(image)
                faces, _ = mtcnn.detect(img)
            except OSError:
                print(f"Truncated image: {image}, skipping")
                pass

            # Appending zero if nothing is found, this is required because the shape is otherwise None which breaks 
            if faces is not None:
                print(f'{faces.shape[0]} faces detected in {newspaper}, {date} on page {page}')
                face_list.append({"newspaper": newspaper, "date": date, "decade": decade, "page": page, "face_count": faces.shape[0]})
            else:
                #print(f"No faces detected on {date} in page {page}")
                face_list.append({"newspaper": newspaper, "date": date, "decade": decade, "page": page, "face_count": 0})
                pass

        df = pd.DataFrame(face_list)
        df.to_csv(f"{self.out_dir}/faces.csv", index=False)

    def process_and_plot(self, csv_path):
        '''
        Computes percentage from the input data and plots this into a two y-axis plot which is saved
        to the out directory.

        Note: Seems the face detection algorithm is not quite conservative enough, because we see
        faces being detected where there are none as seen from a visual inspection.
        '''
        df = pd.read_csv(csv_path)
        df['percent'] = df.groupby(["newspaper",'decade'])['face_count'].transform(
            lambda x: (x > 0).sum() / len(x) * 100
        )

        #grouping by newspaper and decade to plot relationship per newspaper
        decade_count = df.groupby(["newspaper", "decade"])["face_count"].sum().reset_index()
        decade_pct = df.groupby(["newspaper", "decade"])["percent"].mean().reset_index()

        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        #Left y, absolute count and color(hue) per newspaper
        sns.lineplot(
            data=decade_count,
            x="decade",
            y="face_count",
            marker="o",
            hue="newspaper",
            ax=ax1,
            errorbar=None,
        )
        ax1.set_ylabel("Absolute Face Count (circles)")
        ax1.tick_params(axis="y")

        #Right y, percentage - twinning from axis1 is the fastest approach
        ax2 = ax1.twinx()
        sns.lineplot(
            data=decade_pct,
            x="decade",
            y="percent",
            marker="s",
            hue="newspaper",
            ax=ax2,
            errorbar=None,
        )
        ax2.set_ylabel("Percentage of pages with faces (squares)")
        ax2.tick_params(axis="y")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax2.get_legend().remove()#removing this to not get double legends

        ax1.set_xlabel("Decade")
        plt.title("Absolute face count and percentage")
        plt.tight_layout()
        fig.savefig(f'{self.out_dir}/face_figure.jpg')

    def runner(self):
        '''
        Helper function to collect all functionality into one call
        '''
        data = self.load_data(self.data_dir)
        self.detect(data)
        self.process_and_plot(f'{self.out_dir}/faces.csv')

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    classifier = FaceDetector(
        data_dir = root_dir / "data",
        out_dir = root_dir / "out"
    )
    classifier.runner()