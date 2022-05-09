"""
Data loader: load your data and/or create augmented data


Minimum viable code snippet:
```python
dl = DataLoader()

```
"""
import os
import pandas as pd
import tensorflow as tf
from typing import Dict
from abstracts import DataLoaderABC


class DataLoader_PT(DataLoaderABC):
  # FIXME: still to do :D
  def __init__(self, config, **kwargs) -> None:
    super().__init__(config)

  def augment_data(self) -> None:
    """
    Create augmented data given a certain random seed, and save to `augment_dir`
    """
    pass

  def data_generator(self) -> None:
    pass



class DataLoader_TF(DataLoaderABC):
  def __init__(self, config, **kwargs) -> None:
    super().__init__(config)

  def data_generator(self) -> None:
    pass

  def augment_data(self) -> None:
    """
    Create augmented data given a certain random seed, and save
    to `self.config.data.augment_dir`.
    """
    loader = self.data_generator()
    amount_to_create = self.labels.shape[0] * self.config["augment_tf"]["augmentations_amount"]
    for _ in range(amount_to_create):
      a = 1
      # create images

    pass



