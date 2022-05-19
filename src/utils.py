"""
Read in yaml, make it a class property, pass properties to the right thing
https://hackersandslackers.com/simplify-your-python-projects-configuration/
"""
import os
import yaml


class Config:
    """"""

    def __init__(self, cfg_loc: str = "artefacts/configs/data_loader.yml") -> None:
        with open(cfg_loc, "r") as b:
            assert os.path.isfile(
                cfg_loc
            ), "Need valid config file path, got {} instead.".format(cfg_loc)
            configs = yaml.safe_load(b)
        self.cfg_loc = cfg_loc
        self.data = configs["data"]
        self.labels = configs["labels"]
        self.augment_params = configs["augment_params"]
        self.training = configs["training"]
        self.experiment = configs["experiment"]
