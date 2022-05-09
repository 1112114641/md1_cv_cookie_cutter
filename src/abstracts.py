"""
Abstracts serve as a base template for classes, allowing to hide away some of the
required logic into the parent (ABC) class, making child classes inherit these boring
aspects.
Interesting aspects, e.g. custom model definitions, will be
```python
def architecture(self, *args, **kwargs)
  pass

which can then be defined on a case by case basis in the children (c.f. polymorphism/
inheritance).


"""
from abc import ABC, abstractmethod
from src.utils import Config
import pandas as pd
import os


class DataLoaderABC(ABC):
  """
  access settings through config.data / config.labels /
  """
  def __init__(self, cfg_loc) -> None:
    self.config = Config(cfg_loc)

  def read_labels(self,label_dir:str=None)->pd.DataFrame:
    """Read labels, split into one hot labels for multi label case."""
    label_dir = label_dir if label_dir else self.label_dir

    if os.path.isile(label_dir)&label_dir.endswith(".csv"):
      label_dir = label_dir if label_dir else self.label_dir
      labels_df = pd.read_csv(label_dir)
      assert ("filename" in labels_df.columns) & ("labels" in labels_df.columns), "Follow example label file structure."

    elif os.path.isile(label_dir)&label_dir.endswith(".txt"):
      labels = []
      with open(label_dir, "r") as banana:
        for line in banana:
          items = line.rstrip().split(",")  # individual items sans end of line
          labels.append([items[0], items[1:]])  # split into file name, list of labels
      labels_df = pd.DataFrame(labels,)
      labels_df.columns = ["filename", "labels"]
    else:
      raise ValueError(
        "label_dir should point to valid label csv-file. Instead got {}."
        .format(label_dir)
      )

    # for the case of multi-label data:
    if isinstance(labels_df["labels"].iloc[0],list):
      (
        labels_df
        .drop('labels', axis=1)
        .join(labels_df.labels.str.join('|')
          .str.get_dummies())
      )
    return labels_df

  @abstractmethod
  def data_generator():
    pass

  @abstractmethod
  def data_set():
    pass

class model():
  def __init__(self, cfg_loc:str) -> None:
      self.config = Config(cfg_loc)

  @abstractmethod
  def load_data(self):
    pass

  @abstractmethod
  def build_model(self):
    pass

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def evaluate(self):
    pass


# define majority of class that is not going to change in
# here (fit, hyperparams, ...), then leave rest of definition to the corresponding
# class file

# create data_loader first
# model def after (easy head splitting)
# rest after