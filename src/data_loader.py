"""
Data loader: load your data and/or create augmented data


Minimum viable code snippet:
```python
dl = DataLoader()

```
"""
import os
import pandas as pd
from abstracts import DataLoaderABC

# import tensorflow as tf


class DataLoader(DataLoaderABC):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

    def data_generator(self) -> None:
        pass

    def augment_data(self) -> None:
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
