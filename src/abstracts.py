"""
Abstracts serve as a base template for classes, allowing to hide away some of the
required logic into the parent (ABC) class, and setting out a template for childrent to
adher to, making child classes inherit these boring
aspects.
Interesting aspects, e.g. custom model definitions, can be
```python
def architecture(self, *args, **kwargs)
  pass

which can then be defined on a case by case basis in the children ("inheritance").
"""
from abc import ABC, abstractmethod
from src.utils import Config


class DataLoaderABC(ABC):
    """
    access settings through config.data / config.labels /
    """

    def __init__(self, cfg_loc) -> None:
        self.config = Config(cfg_loc)

    @abstractmethod
    def read_labels(self):
        pass

    @abstractmethod
    def data_generator(self):
        """Generator for training"""
        pass

    @abstractmethod
    def persist_data_set(self):
        """Create (augmented) data set and persist."""
        pass


class BaseModelABC(ABC):
    def __init__(self, cfg_loc: str) -> None:
        self.config = Config(cfg_loc)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def model_build(self):
        pass

    @abstractmethod
    def model_train(self):
        pass

    @abstractmethod
    def model_evaluate(self):
        pass

    @abstractmethod
    def model_load_weights(self):
        pass


class AugmentABC(ABC):
    def __init__(self, cfg_loc: str) -> None:
        self.config = Config(cfg_loc)

    # @abstractmethod
    # def simple_augment(self):
    #     pass

    # @abstractmethod
    # def complex_augment(self):
    #     pass

    # @abstractmethod
    # def model_train(self):
    #     pass

    # @abstractmethod
    # def model_evaluate(self):
    #     pass

    # @abstractmethod
    # def model_load_weights(self):
    #     pass


# define majority of class that is not going to change in
# here (fit, hyperparams, ...), then leave rest of definition to the corresponding
# class file

# create data_loader first
# model def after (easy head splitting)
# rest after
