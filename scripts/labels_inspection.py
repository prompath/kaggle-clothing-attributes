# %% [markdown]
# # Libraries

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import PIL.Image as pil_image
import seaborn as sns

from kaggle_clothing_attributes.config import load_config
from kaggle_clothing_attributes.consts import LABEL_MAPPING, COLOR_MAPPING
from kaggle_clothing_attributes.preprocess import load_labels_from_directory
from kaggle_clothing_attributes.utils import show_image_with_labels

# %% [markdown]
# # Constants

# %%
CONFIG = load_config()
images_dir = Path(CONFIG["dataset"]["images_dir"])
labels_dir = Path(CONFIG["dataset"]["labels_dir"])

# %% [markdown]
# # Labels Inspection

# %%
labels_df = load_labels_from_directory(labels_dir)
labels_df = labels_df.reindex(columns=LABEL_MAPPING.keys())

# %%
labels_df.head()

# %%
print(f"Shape of the label dataframe: {labels_df.shape}")

# %%
# Let's check the number of the original image.
images = sorted(list((images_dir).glob("*.jpg")))  # Glob doesn't sort by file name.
print(f"Number of images: {len(images)}")

# %%
idx = 0
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %%
idx = 20
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %%
idx = 100
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %%
idx = 200
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %%
idx = 354
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %%
idx = 900
image_array = pil_image.open(images[idx])
show_image_with_labels(image_array, labels_df, idx)

# %% [markdown]
# The labels are correct given the chosen images. Next, let's look at the distribution of the labels.

# %% [markdown]
# # Label Distribution

# %%
mapped_labels_df = labels_df.apply(
    lambda x: x.map(LABEL_MAPPING[x.name]), axis=0
).replace({None: "Unknown"})
n_images = labels_df.shape[0]


# %%
def plot_label_distribution(label_name: str) -> None:
    """
    Plots the distribution of a label.

    Parameters:
        label_name (str): The name of the label.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        y=mapped_labels_df[label_name].value_counts().index,
        x=mapped_labels_df[label_name].value_counts(normalize=True).values,
        hue=mapped_labels_df[label_name].value_counts().index,
        orient="h",
        palette=dict(COLOR_MAPPING),
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for c in ax.containers:
        ax.bar_label(c, fmt=lambda x: f"{x * 100: 0.1f}%")
    plt.show()


# %%
for label in LABEL_MAPPING.keys():
    plot_label_distribution(label)

# %% [markdown]
# # Notes
# - Found a sample with multiple subjects. There might be more.
#   - Index: 354
# - Some labels have high amount of unknown values.
#   - Label: "category" at 40.5%, "neckline" at 29.2%, and "pattern_solid" at 19.6%.
