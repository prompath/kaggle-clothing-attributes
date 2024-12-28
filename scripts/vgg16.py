# %% [markdown]
# # Libraries

# %%
import tensorflow as tf
import keras
from keras import layers

from kaggle_clothing_attributes.config import load_config
from kaggle_clothing_attributes.preprocess import load_image

# %% [markdown]
# # Constants

# %%
AUTOTUNE = tf.data.AUTOTUNE
CONFIG = load_config()

# %% [markdown]
# # Prepare Datasets

# %%
# train_dataset = tf.data.Dataset.load(CONFIG["dataset"]["train_path"])
# validation_dataset = tf.data.Dataset.load(CONFIG["dataset"]["validation_path"])
# test_dataset = tf.data.Dataset.load(CONFIG["dataset"]["test_path"])
train_dataset = tf.data.Dataset.load("../data/processed/placket_train_records")
validation_dataset = tf.data.Dataset.load(
    "../data/processed/placket_validation_records"
)
test_dataset = tf.data.Dataset.load("../data/processed/placket_test_records")

# %%
for i in train_dataset.take(1):
    print(i)

# %%
target_height, target_width = 300, 300
train_dataset = train_dataset.map(
    lambda x, y: (load_image(x, target_size=(target_height, target_width)), y),
    num_parallel_calls=AUTOTUNE,
)
validation_dataset = validation_dataset.map(
    lambda x, y: (load_image(x, target_size=(target_height, target_width)), y),
    num_parallel_calls=AUTOTUNE,
)
test_dataset = test_dataset.map(
    lambda x, y: (load_image(x, target_size=(target_height, target_width)), y),
    num_parallel_calls=AUTOTUNE,
)

# %%
batch_size = 64
train_dataset = train_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# %% [markdown]
# # VGG16

# %%
conv_base = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
conv_base.trainable = False

# %%
# !!! INCORRECT SETUP. FOR TRIAL AND ERROR ONLY !!!
inputs = keras.Input(shape=(target_height, target_width, 3))
x = keras.applications.vgg16.preprocess_input(inputs)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# %%
model.summary()

# %%
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# %%
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
)
