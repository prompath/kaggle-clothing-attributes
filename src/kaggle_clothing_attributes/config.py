import pytoml

DEFAULT_CONFIG_PATH = "../configs/config.toml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    """
    Loads the configuration from a TOML file.

    Args:
        config_path (str): The path to the configuration file. Defaults to '../configs/config.toml'.

    Returns:
        dict: The loaded configuration.
    """
    with open(config_path) as f:
        config = pytoml.load(f)
    return config
