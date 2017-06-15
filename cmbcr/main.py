"""
Command execution, input/output
"""


import argparse
import yaml
import logging
import os
from pprint import pprint




def load_config_file(filename):
    with open(filename) as f:
        config_doc = yaml.load(f)
    return config_doc




def main():
    from .cr_system import CrSystem

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    config = load_config_file(parsed_args.config_file)


    return 0
