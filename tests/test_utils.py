import pytest
from src.utils import Config


@pytest.mark.parametrize(
    "example_config_loc, example_dict_labels, example_dict_aug_params",
    [
        (
            "tests/artefacts/example_config.yaml",
            {"label_dir": 64, "multi_label": False},
            {"random_seed": 42, "flip_updown": True},
        ),
    ],
)
def testConfig(example_config_loc, example_dict_labels, example_dict_aug_params):
    _testConfig = Config(example_config_loc)
    assert _testConfig.labels == example_dict_labels
    assert _testConfig.augment_params == example_dict_aug_params
