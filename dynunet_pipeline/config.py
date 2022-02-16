import yaml

def get_config(file):
    f = open(file)
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config
