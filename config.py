import yaml
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(os.path.join(PROJECT_PATH, "config.py"))