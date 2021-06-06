import yaml
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(PROJECT_PATH, "config.yaml"), "r"), Loader=yaml.FullLoader)