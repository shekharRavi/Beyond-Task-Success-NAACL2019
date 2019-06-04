import json

def load_config(config_file):
    with open(config_file, 'r') as f_config:
        config = json.loads(f_config.read())

    return config
