"""
This module implements the preprocessing steps for the Clothing Attributes dataset.
"""

import logging
from pathlib import Path

import pandas as pd
import scipy.io as sio

from .config import load_config

CONFIG = load_config()
log_name = Path(__file__).stem
logger = logging.getLogger(log_name)
logging.basicConfig(
    filename=Path(CONFIG["logging"]["log_dir"]) / f"{log_name}.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


def load_labels_from_directory(labels_dir: str) -> pd.DataFrame:
    """
    Loads labels from a directory of mat files and returns them as a pandas DataFrame.
    The data type of the labels is converted to Int8.

    Parameters:
        label_dir (str): The path to the directory containing the mat files.

    Returns:
        pd.DataFrame: A DataFrame with the labels, where each column corresponds to a file name in the directory.
    """
    label_files = list(Path(labels_dir).glob("*.mat"))
    logger.info(f"Number of label files found: {len(label_files)}")
    label_df = pd.DataFrame()
    for label_file in label_files:
        label_mat = sio.loadmat(label_file)
        label_array = label_mat["GT"]
        label_title = label_file.stem.rstrip("_GT")
        label_df[label_title] = label_array.flatten()
    logger.info(f"Shape of the label dataframe: {label_df.shape}")
    label_df = label_df.astype("Int8")
    logger.info("Casted the data type of the dataframe to Int8")
    return label_df
