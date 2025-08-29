import yaml
import os

class Config:
    def __init__(self, config_data: dict):
        for key, value in config_data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

def load_config(config_path: str) -> Config:
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at path: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        
    return Config(config_data)
