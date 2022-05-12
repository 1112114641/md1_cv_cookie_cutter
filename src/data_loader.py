"""
Data loader: load your data and/or create augmented data


Minimum viable code snippet:
```python
dl = DataLoader()

```
"""
import os
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
from typing import Tuple
from abstracts import DataLoaderABC
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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
        self._train_test_split_labels()

    def data_generator(self, *args) -> Tuple[tf.data.Dataset]:
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
            image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
            # image = tf.py_function(func=augment_images, inp=[image], Tout=tf.uint8)
            return image

        images_ds = filenames_ds.map(
            _parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # FIXME: check one-hot requirement for multi label? Object Detection?
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

    def _train_test_split_labels(
        self,
    ) -> Tuple[pd.DataFrame]:
        """
        Returns indices of stratified train/test split set.
        If `self.config.labels["val_test_subsplits"]`, ensure each item occurs with a
        sufficient frequency.
        Caution: does not filter for duplicate files in the file list!

        c.f. https://github.com/trent-b/iterative-stratification
        """
        mlsss_train_valtest = MultilabelStratifiedShuffleSplit(
            n_splits=2,
            train_size=(1 - self.config.labels["train_test_split"]),
            test_size=self.config.labels["train_test_split"],
            random_state=self.config.labels["random_seed"],
        )

        try:
            Y = self.labels.labels  # labels to be stratified,
            X = np.zeros(self.labels.shape[0])  # placeholder
            if self.config.labels["train_test_split"] > 0:
                train_indices, valid_indices = next(mlsss_train_valtest.split(X, Y))
                self.train_labels = self.labels[train_indices]
                self.valid_labels = self.labels[valid_indices]

                if self.config.labels["val_test_subsplits"] > 0:
                    mlsss_val_test = MultilabelStratifiedShuffleSplit(
                        n_splits=2,
                        train_size=(1 - self.config.labels["val_test_subsplits"]),
                        test_size=self.config.labels["val_test_subsplits"],
                        random_state=self.config.labels["random_seed"],
                    )
                    valid_indices, test_indices = next(
                        mlsss_val_test.split(
                            np.zeros(self.valid_labels.shape[0]),
                            self.valid_labels.labels,
                        )
                    )
                    self.valid_labels = self.labels[valid_indices]
                    self.test_labels = self.labels[test_indices]
        except IndexError:
            X_train, X_valid, y_train, y_valid = train_test_split(
                np.zeros((self.labels.shape[0],)),
                self.labels.labels,
                random_state=self.config.labels["random_seed"],
                test_size=self.config.labels["train_test_split"],
                train_size=(1 - self.config.labels["train_test_split"]),
                stratify=self.labels.labels,
            )
            self.train_labels = self.labels[y_train.index]
            self.valid_labels = self.labels[y_valid.index]
            if self.config.labels["val_test_subsplits"] > 0:
                X_valid, X_test, y_valid, y_test = train_test_split(
                    np.zeros(self.valid_labels.shape[0]),
                    self.valid_labels.labels,
                    random_state=self.config.labels["random_seed"],
                    test_size=self.config.labels["val_test_subsplits"],
                    train_size=(1 - self.config.labels["val_test_subsplits"]),
                    stratify=self.labels.labels,
                )
                self.valid_labels = self.labels[y_valid.index]
                self.test_labels = self.labels[y_test.index]

        return (
            self.train_labels,
            self.valid_labels,
            self.test_labels,
        )

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
        """
        Read labels, split into one hot labels for multi label case.
        For classification task, labels are expected to come in the column format:
         filename, label1, label2, label3, ...
        For object detection the format should be:
         filename, label1, h1, w1, x1, y1, label2, h2, w2, x2, y2, ...
        Both txt/csv files are expected to have these right header structures.

          Returns:
            pd.DataFrame: filenames, labels (and bounding boxes) in the forma split up
            into columns into columns like:
            fname:str, labels:List[str], bbox_hwxy:List[List[int,int,int,int]]
        """
        label_dir = label_dir if label_dir else self.label_dir

        if os.path.isile(label_dir):
            label_dir = label_dir if label_dir else self.label_dir
            labels_df = pd.read_csv(label_dir).fillna("")

            # fix for object detection bboxes
            if labels_df.shape[1] >= 6:

                def _reshape_label_df_for_object_detect(
                    labels: pd.DateFrame,
                ) -> pd.DataFrame:
                    max_objects = (labels.shape[1] - 1) / 5
                    labels_tmp = labels["filename"]

                    # determine label / bbox col positions
                    label_cols = [n for n in range(max_objects) if ((n - 1) % 5 == 0)]
                    bbox_cols = [
                        n
                        for n in range(max_objects)
                        if (
                            ((n - 2) % 5 == 0)
                            | ((n - 3) % 5 == 0)
                            | ((n - 4) % 5 == 0)
                            | ((n - 5) % 5 == 0) & (n > 2)
                        )
                    ]
                    # all labels per row added to list
                    labels_tmp["labels"] = labels.iloc[:, label_cols].values.tolist()
                    # filter out empty labels
                    labels_tmp["labels"] = labels_tmp["labels"].apply(
                        lambda x: [i for i in x if i != ""]
                    )
                    # create one column of all bounding box hwxy values
                    labels_tmp["bbox_hwxy"] = labels.iloc[:, bbox_cols].values.tolist()
                    labels_tmp["bbox_hwxy"] = labels_tmp["bbox_hwxy"].apply(
                        # grab every 4th element, add that one and following four to a list
                        lambda x: [
                            x[i : i + 4]
                            if (x[i] != "")
                            & (x[i + 1] != "")
                            & (x[i + 2] != "")
                            & (x[i + 3] != "")
                            else list()
                            for i in range(0, len(x), 4)
                        ]
                    )
                    # filter out empty bbox elements
                    labels_tmp["bbox_hwxy"] = labels_tmp["bbox_hwxy"].apply(
                        lambda x: [i for i in x if i != ""]
                    )

                    # sanity check to remove empty labels/bbox:
                    faulty = labels_tmp[
                        labels_tmp["labels"].map(lambda x: len(x)) == 0
                    ].index.values
                    assert (
                        len(faulty) == 0
                    ), "Faulty label data/labels missing. Check rows {}.".format(faulty)
                    # same to ensure each label has a bbox
                    tmp_screen_labels = labels_tmp["labels"].apply(lambda x: len(x))
                    tmp_screen_bbox = labels_tmp["bbox_hwxy"].apply(lambda x: len(x))
                    abc = np.where(
                        tmp_screen_labels == tmp_screen_bbox,
                        np.zeros((tmp_screen_bbox.shape[0],)),
                        np.zeros((tmp_screen_bbox.shape[0],)),
                    )
                    assert (
                        abc.sum() == 0
                    ), "Not enough bounding boxes/labels in rows: {}".format(
                        abc[abc != 0]
                    )
                    # and filter out empty rows w/o labels
                    return labels_tmp

                labels_df = _reshape_label_df_for_object_detect(labels=labels_df)
            assert ("filename" in labels_df.columns) & (
                "labels" in labels_df.columns
            ), "Follow example label file structure!"
        else:
            logging.info(
                "label_dir should point to valid label txt/csv-file. Instead got {}.".format(
                    label_dir
                )
            )
            raise ValueError(
                "label_dir should point to valid label txt/csv-file. Instead got {}.".format(
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
