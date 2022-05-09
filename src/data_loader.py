"""
Data loader: load your data and/or create augmented data


Minimum viable code snippet:
```python
dl = DataLoader()

```
"""
import os
import logging
import pandas as pd
import tensorflow as tf
from typing import Tuple
from abstracts import DataLoaderABC

# import joblib
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE


class DataLoader(DataLoaderABC):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)
        self.labels = self.read_labels(self.config.labels["label_dir"])

    def data_generator(self) -> Tuple[tf.data.Dataset]:
        """Create data ready for training."""
        # image_filenames = [
        # os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames
        # if f.endswith("jpg") or f.endswith("png")
        # ]
        image_filenames = self.labels.filename
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_filenames)

        def _parse_image(filename: str) -> tf.Tensor:
            """
            Read in image data. Normalisation to be performed by the model at train/
            inference run-time.
            """
            try:
                image = tf.io.read_file(self.config.data["data_dir"] + filename)
            except FileNotFoundError:
                logging.info(f"{filename} not found in {self.config.data['data_dir']}.")
            image = tf.io.decode_image(image, channels=3)
            # image = tf.image.rgb_to_grayscale(image)
            img_size = self.config.data["img_height"]
            image = tf.image.resize(image, size=[img_size, img_size])
            image = tf.image.convert_image_dtype(image, tf.uint8)
            # image = tf.py_function(func=augment_images, inp=[image], Tout=tf.uint8)
            return image

        images_ds = filenames_ds.map(
            _parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # FIXME: check one-hot requirement for multi label?
        labels_ds = tf.data.Dataset.from_tensor_slices(self.labels.labels)
        ds = tf.data.Dataset.zip((images_ds, labels_ds))

        def configure_for_performance(ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
            """
            Create train/val/test split, augment train, and optimise for fast data
            loading.
            """
            val_split = self.config.labels["train_test_split"]
            test_split = self.config.labels["val_test_subsplits"]
            no_of_images = self.labels.shape[0]
            ds_valid = ds.take(int(val_split * no_of_images))
            ds_valid = ds_valid.batch(BATCH_SIZE)
            ds_valid = ds_valid.prefetch(buffer_size=AUTOTUNE)
            ds_test = ds.skip(int(val_split * no_of_images))
            ds_test = ds_test.take(int(test_split * no_of_images))
            ds_test = ds_test.batch(BATCH_SIZE)
            ds_train = ds.skip(
                int(val_split * no_of_images) + int(test_split * no_of_images)
            )
            ds_train = ds_train.shuffle(buffer_size=10000)
            ds_train = ds_train.batch(BATCH_SIZE).repeat()
            ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
            return ds_train, ds_valid, ds_test

        ds_train, ds_valid, ds_test = configure_for_performance(ds)
        return ds_train, ds_valid, ds_test

    def persist_data_set(self) -> None:
        """
        Create augmented data given a certain random seed, and save to
        `DataLoader.config.data.augment_dir`.
        """
        loader = self.data_generator()
        loader = loader
        amount_to_create = (
            self.labels.shape[0] * self.config["augment_tf"]["augmentations_amount"]
        )
        for _ in range(amount_to_create):
            a = 1
            a = a
            # create images
        pass

    def read_labels(self, label_dir: str = None) -> pd.DataFrame:
        """Read labels, split into one hot labels for multi label case."""
        label_dir = label_dir if label_dir else self.label_dir

        if os.path.isile(label_dir) & label_dir.endswith(".csv"):
            label_dir = label_dir if label_dir else self.label_dir
            labels_df = pd.read_csv(label_dir)
            assert ("filename" in labels_df.columns) & (
                "labels" in labels_df.columns
            ), "Follow example label file structure."

        elif os.path.isile(label_dir) & label_dir.endswith(".txt"):
            labels = []
            with open(label_dir, "r") as banana:
                for line in banana:
                    items = line.rstrip().split(
                        ","
                    )  # individual items sans end of line
                    labels.append(
                        [items[0], items[1:]]
                    )  # split into file name, list of labels
            labels_df = pd.DataFrame(
                labels,
            )
            labels_df.columns = ["filename", "labels"]
        else:
            raise ValueError(
                "label_dir should point to valid label csv-file. Instead got {}.".format(
                    label_dir
                )
            )

        # for the case of multi-label data:
        if isinstance(labels_df["labels"].iloc[0], list):
            (
                labels_df.drop("labels", axis=1).join(
                    labels_df.labels.str.join("|").str.get_dummies()
                )
            )
        return labels_df


if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/data_loader.log",
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
