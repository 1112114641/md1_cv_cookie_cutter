"""
Data loader: load your data and/or create augmented data, which can then be persisted.

TODO: implement the augmentations as part of the pipeline
TODO: implement persist data fct
"""
import os
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
from typing import Tuple  # , Callable, Dict, List
from abstracts import DataLoaderABC
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# from src.augmentations import Augment
# import joblib
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder

BUFFER_SIZE = 5000
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE


class DataLoader(DataLoaderABC):
    """
    Data Loader for a tensorflow/image classification setup. Needs to be properly
    extended to object detection purposes.

    FIXME: extend to object detection tasks & pytorch
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)
        self.labels = self.read_labels(self.config.labels_dir)
        self._train_test_split_labels()

    def data_generator(
        self,
    ) -> Tuple[tf.data.Dataset]:
        """Create data ready for training."""
        # image_filenames = [
        # os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames
        # if f.endswith("jpg") or f.endswith("png")
        # ]

        # load image names and labels
        # FIXME: fix for object detection
        # FIXME: quite a lot of repetition here
        def _read_data_zip_labels(
            labels: pd.DataFrame, fnames: pd.DataFrame
        ) -> tf.data.Dataset:
            fnames = tf.data.Dataset.from_tensor_slices(fnames)
            images = fnames.map(self._parse_image, num_parallel_calls=AUTOTUNE)
            labels = tf.data.Dataset.from_tensor_slices(labels)
            return tf.data.Dataset.zip((images, labels))

        ds_train = _read_data_zip_labels(
            labels=self.train_labels.labels, fname=self.train_labels.filename
        )
        ds_valid = _read_data_zip_labels(
            labels=self.valid_labels.labels, fname=self.valid_labels.filename
        )
        ds__test = _read_data_zip_labels(
            labels=self.test_labels.labels, fname=self.test_labels.filename
        )

        # ds_train = self.augment.run_augmentations()

        ds_train, ds_valid, ds_test = self._configure_for_performance(
            valid=ds_valid, test=ds__test, train=ds_train
        )
        return ds_train, ds_valid, ds_test

    def persist_data_set(self) -> None:
        """
        Create augmented data given a certain random seed, and save to
        `DataLoader.config.data.augment_dir`.
        """
        # loader = self.data_generator()
        # loader = loader
        # amount_to_create = (
        #     self.labels.shape[0] * self.config.augment_params_augmentations_amount
        # )
        # for _ in range(amount_to_create):
        #     a = 1
        #     a = a
        #     # create images
        pass

    def _train_test_split_labels(
        self,
    ) -> Tuple[pd.DataFrame]:
        """
        Returns indices of stratified train/test split set.
        If `self.config.labels_val_test_subsplits`, ensure each item occurs with a
        sufficient frequency.
        Caution: does not filter for duplicate files in the file list!

        c.f. https://github.com/trent-b/iterative-stratification
        """
        mlsss_train_valtest = MultilabelStratifiedShuffleSplit(
            n_splits=2,
            train_size=(1 - self.config.labels_train_test_split),
            test_size=self.config.labels_train_test_split,
            random_state=self.config.random_seed,
        )

        try:
            Y = self.labels.labels  # labels to be stratified,
            X = np.zeros(self.labels.shape[0])  # placeholder
            if self.config.labels_train_test_split > 0:
                train_indices, valid_indices = next(mlsss_train_valtest.split(X, Y))
                self.train_labels = self.labels[train_indices]
                self.valid_labels = self.labels[valid_indices]

                if self.config.labels_val_test_subsplits > 0:
                    mlsss_val_test = MultilabelStratifiedShuffleSplit(
                        n_splits=2,
                        train_size=(1 - self.config.labels_val_test_subsplits),
                        test_size=self.config.labels_val_test_subsplits,
                        random_state=self.config.random_seed,
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
                random_state=self.config.random_seed,
                test_size=self.config.labels_train_test_split,
                train_size=(1 - self.config.labels_train_test_split),
                stratify=self.labels.labels,
            )
            self.train_labels = self.labels[y_train.index]
            self.valid_labels = self.labels[y_valid.index]
            if self.config.labels_val_test_subsplits > 0:
                X_valid, X_test, y_valid, y_test = train_test_split(
                    np.zeros(self.valid_labels.shape[0]),
                    self.valid_labels.labels,
                    random_state=self.config.random_seed,
                    test_size=self.config.labels_val_test_subsplits,
                    train_size=(1 - self.config.labels_val_test_subsplits),
                    stratify=self.labels.labels,
                )
                self.valid_labels = self.labels[y_valid.index]
                self.test_labels = self.labels[y_test.index]
            else:
                self.test_labels = pd.DataFrame(columns=["filename"])

        return (
            self.train_labels,
            self.valid_labels,
            self.test_labels,
        )

    def _parse_image(self, filename: str) -> tf.Tensor:
        """
        Read in image data. Normalisation to be performed by the model at train/
        inference run-time.
        """
        try:
            image = tf.io.read_file(self.config.data_dir + filename)
        except FileNotFoundError:
            logging.info(f"{filename} not found in {self.config.data_dir}.")
        image = tf.io.decode_image(image, channels=3)
        # image = tf.image.rgb_to_grayscale(image)
        img_size = self.config.data["img_height"]
        image = tf.image.resize(image, size=[img_size, img_size])
        image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
        # image = tf.py_function(func=augment_images, inp=[image], Tout=tf.uint8)
        return image

    # def _augment_complex(
    #     self, ds: tf.data.Dataset, ds2: tf.data.Dataset
    # ) -> tf.data.Dataset:
    #     if self.config.augment_params_cutmix:
    #         ds, ds2 = cutmix(  # FIXME: fix for object detection
    #             ds, ds2, max_size=30, seed=self.config.random_seed
    #         )
    #     if self.config.augment_params_mixup:
    #         ds, ds2 = mixup(  # FIXME: fix for object detection
    #             ds, ds2, seed=self.config.random_seed
    #         )
    #     return ds

    def _configure_for_performance(
        self, train: tf.data.Dataset, valid: tf.data.Dataset, test: tf.data.Dataset
    ) -> tf.data.Dataset:
        """
        Create train/val/test split, augment train, and optimise for fast data
        loading.
        """
        train = (
            train.shuffle(buffer_size=BUFFER_SIZE)
            .batch(self.config.training_batch_size)
            .repeat()
            .prefetch(buffer_size=AUTOTUNE)
        )
        valid = valid.batch(self.config.training_batch_size).prefetch(
            buffer_size=AUTOTUNE
        )
        test = test.batch(self.config.training_batch_size).prefetch(
            buffer_size=AUTOTUNE
        )
        return train, valid, test

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

        # # for the case of multi-label data:
        # if isinstance(labels_df["labels"].iloc[0], list):
        #   (
        #     labels_df.drop("labels", axis=1).join(
        #       labels_df.labels.str.join("|").str.get_dummies()
        #     )
        #   )
        return labels_df


if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/data_loader.log",
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
