import matplotlib.pyplot as plt
import pandas as pd
from PIL.Image import Image

from kaggle_clothing_attributes.consts import LABEL_MAPPING


def show_image_with_labels(image_array: Image, labels_df: pd.DataFrame, idx: int):
    """
    Displays an image with its corresponding labels.

    Parameters:
        image_array (np.ndarray): The image array.
        label_df (pd.DataFrame): The DataFrame containing the labels.
        idx (int): The index of the image in the DataFrame.
    """
    plt.imshow(image_array)
    plt.axis("off")
    plt.show()
    for col in labels_df.columns:
        mapped_value = LABEL_MAPPING[col].get(labels_df.loc[idx, col])
        if pd.isnull(mapped_value):
            mapped_value = "Unknown"
        print(f"{' '.join(col.split('_')).title()}: {mapped_value}")
