"""
Read in yaml, make it a class property, pass properties to the right thing
"""
import os
import yaml


class Config:
    def __init__(self, cfg_loc: str = "artefacts/configs/data_loader.yml") -> None:
        with open(cfg_loc, "r") as b:
            assert os.path.isfile(
                cfg_loc
            ), "Need valid config file path, got {} instead.".format(cfg_loc)

            configs = yaml.safe_load(b)

        self.cfg_loc = cfg_loc

        for key in configs.keys():
            setattr(self, key, configs[key])
